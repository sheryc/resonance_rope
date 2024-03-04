from typing import Any, Dict

import plotly.graph_objects as go
import torch
from aim import Text, Figure
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric, Accuracy

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AlgorithmicDecoderModule(LightningModule):
    def __init__(self, model, modular_num, optimizer, scheduler, generation_config, compile=False,
                 train_max_length=64, val_test_max_length=256, initial_token_count=4, acc_every=20):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['model', 'generation_config'])
        self.model = model
        self.generation_config = generation_config(eos_token_id=modular_num + 4, pad_token_id=modular_num + 1,
                                                   use_cache=True)

        self.eval_loss_fct = torch.nn.CrossEntropyLoss()

        self.val_acc_id = Accuracy(task='multiclass', num_classes=modular_num + 5)
        self.val_acc_ood = Accuracy(task='multiclass', num_classes=modular_num + 5)
        self.test_acc_id = Accuracy(task='multiclass', num_classes=modular_num + 5)
        self.test_acc_ood = Accuracy(task='multiclass', num_classes=modular_num + 5)
        for i in range(0, val_test_max_length + 1, acc_every):
            if i == 0:
                continue
            setattr(self, f'val_acc_digit{i}', Accuracy(task='multiclass', num_classes=modular_num + 5))
            setattr(self, f'test_acc_digit{i}', Accuracy(task='multiclass', num_classes=modular_num + 5))
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_id_best = MaxMetric()
        self.val_acc_ood_best = MaxMetric()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, past_key_values=None,
                use_cache=None, output_attentions=None, return_dict=None, lengths=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,
                          past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
                          return_dict=return_dict, lengths=lengths)

    def on_train_start(self) -> None:
        for metric in [self.val_loss, self.test_loss, self.val_acc_id, self.val_acc_ood, self.test_acc_id,
                       self.test_acc_ood, self.val_acc_id_best, self.val_acc_ood_best]:
            metric.reset()
        for i in range(0, self.hparams.val_test_max_length + 1, self.hparams.acc_every):
            if i == 0:
                continue
            getattr(self, f'val_acc_digit{i}').reset()

    def training_step(self, batch, batch_idx):
        lengths = batch.get('lengths', None)
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'],
                       use_cache=True, output_attentions=False, return_dict=True, lengths=lengths)
        loss = outputs['loss']
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, generate_seq, gt_seq, complete_generate_seq, _ = self.generate_for_eval_test(batch, batch_idx, mode='val')
        self.val_loss(loss)
        self.val_acc_id(generate_seq[:, :self.hparams.train_max_length - self.hparams.initial_token_count].reshape(-1),
                        gt_seq[:, :self.hparams.train_max_length - self.hparams.initial_token_count].reshape(-1))
        self.val_acc_ood(generate_seq[:, self.hparams.train_max_length - self.hparams.initial_token_count:].reshape(-1),
                         gt_seq[:, self.hparams.train_max_length - self.hparams.initial_token_count:].reshape(-1))
        for i in range(0, self.hparams.val_test_max_length + 1, self.hparams.acc_every):
            if i == 0:
                continue
            if i - self.hparams.initial_token_count <= generate_seq.shape[-1]:
                getattr(self, f'val_acc_digit{i}')(
                    generate_seq[:, i - self.hparams.initial_token_count - 1].reshape(-1),
                    gt_seq[:, i - self.hparams.initial_token_count - 1].reshape(-1))
                self.log(f'val/acc_digit{i}', getattr(self, f'val_acc_digit{i}'),
                         on_step=False, on_epoch=True, prog_bar=False)
        if batch_idx == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'name') and 'aim' in self.logger.name:
            generate_seq = [str(token) for token in complete_generate_seq.tolist()]
            gt_seq = [str(token) for token in batch['labels'].squeeze(0).tolist()]
            self.logger.experiment.track(Text(' '.join(generate_seq)), name='val/generate_text', step=self.global_step)
            self.logger.experiment.track(Text(' '.join(gt_seq)), name='val/gt_text', step=self.global_step)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_id", self.val_acc_id, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_ood", self.val_acc_ood, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, generate_seq, gt_seq, complete_generate_seq, attention_patterns = self.generate_for_eval_test(batch,
                                                                                                            batch_idx,
                                                                                                            mode='test')
        self.test_loss(loss)
        self.test_acc_id(generate_seq[:, :self.hparams.train_max_length - self.hparams.initial_token_count].reshape(-1),
                         gt_seq[:, :self.hparams.train_max_length - self.hparams.initial_token_count].reshape(-1))
        self.test_acc_ood(
            generate_seq[:, self.hparams.train_max_length - self.hparams.initial_token_count:].reshape(-1),
            gt_seq[:, self.hparams.train_max_length - self.hparams.initial_token_count:].reshape(-1))
        for i in range(0, self.hparams.val_test_max_length + 1, self.hparams.acc_every):
            if i == 0:
                continue
            if i - self.hparams.initial_token_count < generate_seq.shape[-1]:
                getattr(self, f'test_acc_digit{i}')(
                    generate_seq[:, i - self.hparams.initial_token_count - 1].reshape(-1),
                    gt_seq[:, i - self.hparams.initial_token_count - 1].reshape(-1))
                self.log(f'test/acc_digit{i}', getattr(self, f'test_acc_digit{i}'),
                         on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_id", self.test_acc_id, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_ood", self.test_acc_ood, on_step=False, on_epoch=True, prog_bar=True)

        # if attention_patterns is not None:
        #     if batch_idx == 0:
        #         try:
        #             for instance_id in range(3):
        #                 for layer_id, layer in enumerate(attention_patterns):
        #                     for head_id in range(layer['probs'].shape[1]):
        #                         pattern = layer['probs'][instance_id, head_id, ...].detach().cpu().numpy()  # l+1, l+1
        #                         fig = go.Figure(data=go.Heatmap(
        #                             z=pattern,
        #                             x=list(range(pattern.shape[0])),
        #                             y=list(range(pattern.shape[1])),
        #                             colorscale='Plasma',
        #                             zmin=0,
        #                             zmax=1
        #                         ))
        #                         fig.update_layout(
        #                             title=f'Attention Pattern for Layer {layer_id}, Head {head_id}, Instance {instance_id}',
        #                             xaxis_title='Input position',
        #                             yaxis_title='Output position',
        #                             yaxis_autorange='reversed'
        #                         )
        #                         self.logger.experiment.track(Figure(fig), name=f'test/attention_l{layer_id}h{head_id}',
        #                                                      step=self.global_step + instance_id)
        #         except Exception:
        #             pass

    def generate_for_eval_test(self, batch, batch_idx, mode='val'):
        if mode == 'val':
            output_attentions = False
        else:
            output_attentions = None
        lengths = batch.get('lengths', None)
        outputs = self(
            input_ids=torch.cat((batch['input_ids'], batch['labels'][:, batch['input_ids'].shape[-1]:]), dim=-1),
            labels=batch['labels'], use_cache=True, output_attentions=output_attentions, return_dict=True,
            lengths=lengths)
        loss = outputs['loss']
        logits = outputs['logits']  # b, l, h
        complete_generate_seq = torch.argmax(logits, dim=-1)
        generate_seq = complete_generate_seq[:, batch['input_ids'].shape[-1] - 1: -1]
        gt_seq = batch['labels'][:, batch['input_ids'].shape[-1]:]
        attention_patterns = outputs.get('attentions', None)
        return loss, generate_seq, gt_seq, complete_generate_seq, attention_patterns

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_acc_id_best(self.val_acc_id.compute())
        self.val_acc_ood_best(self.val_acc_ood.compute())
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_id_best", self.val_acc_id_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_ood_best", self.val_acc_ood_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer, total_steps=self.trainer.estimated_stepping_batches)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
