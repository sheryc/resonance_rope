import os
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.multiprocessing import set_sharing_strategy
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from src.data.components import (
    RecurrentSequenceDataset,
    SemiRecurrentSequenceDataset,
    CotSequenceDataset,
    DataCollatorForSeq2SeqRNNs)
from src.utils import create_directories_for_path, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


#
def set_worker_sharing_strategy(worker_id: int) -> None:
    # This solves the bug of "Too many open files" when the batch_size / num_worker is large
    # https://github.com/pytorch/pytorch/issues/11201
    set_sharing_strategy('file_system')


class AlgorithmicSeqDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = 'data/',
            data_type: str = 'recurrent',
            modular: int = 113,
            near_count: int = 2,
            far_count: int = 2,
            seq_len_train: int = 64,
            seq_len_val_test_multiplier: int = 4,
            train_size: int = 1000,
            val_size: int = 1000,
            test_size: int = 1000,
            train_batch_size: int = 64,
            val_batch_size: int = 64,
            test_batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            arith_task: str = 'mod_addition',
            model_mode='transformers',
    ) -> None:
        super().__init__()

        data_classes = [RecurrentSequenceDataset, SemiRecurrentSequenceDataset, CotSequenceDataset]

        assert data_type in [cl.dataset_name() for cl in data_classes], 'Unknown data type.'
        assert model_mode in ['transformers', 'rnns'], 'Unknown model mode.'

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data = dict()

        for cl in data_classes:
            if data_type == cl.dataset_name():
                self.data_class = cl

        self.collator = DataCollatorForSeq2Seq

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        hparams = self.hparams
        folder = os.path.join(hparams.data_dir, hparams.data_type,
                              f'mod{hparams.modular}_n{hparams.near_count}f{hparams.far_count}')
        if not os.path.exists(folder):
            log.info(f'Preparing data in {folder}...')
            data = dict()
            data['train'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                            seq_len=hparams.seq_len_train,
                                            train_size=hparams.train_size, val_size=hparams.val_size,
                                            test_size=hparams.test_size,
                                            near_count=hparams.near_count, far_count=hparams.far_count,
                                            current_split='train', model_mode=hparams.model_mode,
                                            arith_task=hparams.arith_task)
            data['val'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                          seq_len=hparams.seq_len_train * hparams.seq_len_val_test_multiplier,
                                          train_size=hparams.train_size, val_size=hparams.val_size,
                                          test_size=hparams.test_size,
                                          near_count=hparams.near_count, far_count=hparams.far_count,
                                          current_split='val', model_mode=hparams.model_mode,
                                          arith_task=hparams.arith_task)
            data['test'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                           seq_len=hparams.seq_len_train * hparams.seq_len_val_test_multiplier,
                                           train_size=hparams.train_size, val_size=hparams.val_size,
                                           test_size=hparams.test_size,
                                           near_count=hparams.near_count, far_count=hparams.far_count,
                                           current_split='test', model_mode=hparams.model_mode,
                                           arith_task=hparams.arith_task)

            for split in ['train', 'val', 'test']:
                # if split == 'train':
                #     len_min, len_max = hparams.seq_len_train_min, hparams.seq_len_train_max
                # else:
                #     len_min, len_max = hparams.seq_len_val_test_min, hparams.seq_len_val_test_max
                filename = data[split].filename
                create_directories_for_path(filename)
                if not os.path.exists(filename):
                    data[split].generate_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        hparams = self.hparams
        if self.data.get('train') is None:
            log.info('Setting up training data...')
            self.data['train'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                                 seq_len=hparams.seq_len_train,
                                                 train_size=hparams.train_size, val_size=hparams.val_size,
                                                 test_size=hparams.test_size,
                                                 near_count=hparams.near_count, far_count=hparams.far_count,
                                                 current_split='train', model_mode=hparams.model_mode,
                                                 arith_task=hparams.arith_task)
            self.data['train'].load_data()
            self.data['train'].data = self.data['train'].data.map(self.data['train'].process_instance, num_proc=1,
                                                                  remove_columns=['text'])
        if self.data.get('val') is None:
            log.info('Setting up eval data...')
            self.data['val'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                               seq_len=hparams.seq_len_train * hparams.seq_len_val_test_multiplier,
                                               train_size=hparams.train_size, val_size=hparams.val_size,
                                               test_size=hparams.test_size,
                                               near_count=hparams.near_count, far_count=hparams.far_count,
                                               current_split='val', model_mode=hparams.model_mode,
                                               arith_task=hparams.arith_task)
            self.data['val'].load_data()
            self.data['val'].data = self.data['val'].data.map(self.data['val'].process_instance, num_proc=1,
                                                              remove_columns=['text'])
        if self.data.get('test') is None:
            log.info('Setting up test data...')
            self.data['test'] = self.data_class(data_dir=hparams.data_dir, modular=hparams.modular,
                                                seq_len=hparams.seq_len_train * hparams.seq_len_val_test_multiplier,
                                                train_size=hparams.train_size, val_size=hparams.val_size,
                                                test_size=hparams.test_size,
                                                near_count=hparams.near_count, far_count=hparams.far_count,
                                                current_split='test', model_mode=hparams.model_mode,
                                                arith_task=hparams.arith_task)
            self.data['test'].load_data()
            self.data['test'].data = self.data['test'].data.map(self.data['test'].process_instance, num_proc=1,
                                                                remove_columns=['text'])
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data['train'],
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collator(self.data['train'].tokenizer, return_tensors='pt'),
            worker_init_fn=set_worker_sharing_strategy
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data['val'],
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator(self.data['val'].tokenizer, return_tensors='pt'),
            worker_init_fn=set_worker_sharing_strategy
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data['test'],
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator(self.data['test'].tokenizer, return_tensors='pt'),
            worker_init_fn=set_worker_sharing_strategy
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = AlgorithmicSeqDataModule()
