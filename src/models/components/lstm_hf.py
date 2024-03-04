# Modified from https://huggingface.co/grostaco/lstm-base

import torch.nn as nn
import torch.utils.data
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from src.models.components.model_config import LSTMConfig


# class LSTMPreTrainedModel(PreTrainedModel):
#     config_class = LSTMConfig
#
#     def _init_weights(self, module: nn.Module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LSTM):
#             for name, param in module.named_parameters():
#                 if 'bias' in name:
#                     nn.init.zeros_(param)
#                 elif 'weight' in name:
#                     nn.init.orthogonal_(param)


class LSTMClassificationHead(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()

        dim_size = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        self.dense = nn.Linear(dim_size, dim_size)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(dim_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class LSTMs(nn.Module):
    def __init__(self, config: LSTMConfig):
        # super().__init__(config)
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_index)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, config.n_layers,
                            bidirectional=config.bidirectional,
                            dropout=config.classifier_dropout, batch_first=True)
        self.dropout = nn.Dropout(config.classifier_dropout)

        self.apply(self._init_weights)

    def forward(self, input_ids: torch.FloatTensor, length: torch.LongTensor, **kwargs):
        embedded = self.dropout(self.embedding(input_ids))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length.cpu(), batch_first=True,
                                                            enforce_sorted=False)

        all_hidden_states, (last_hidden, _) = self.lstm(packed_embedded)

        all_hidden_states = nn.utils.rnn.pad_packed_sequence(all_hidden_states, batch_first=True)[0]

        if self.lstm.bidirectional:
            last_hidden_state = torch.cat(
                [last_hidden[-1], last_hidden[-2]], dim=-1)
        else:
            last_hidden_state = last_hidden[-1]
        return all_hidden_states, last_hidden_state, last_hidden

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)


class LSTMForCausalLM(nn.Module):
    def __init__(self, config: LSTMConfig):
        # super().__init__(config)
        super().__init__()
        self.lstm = LSTMs(config)
        self.classification_head = LSTMClassificationHead(config)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, lengths: torch.LongTensor, labels=None, **kwargs):
        all_hidden_states, last_hidden_state, last_hidden = self.lstm(input_ids, lengths, **kwargs)
        lm_logits = self.classification_head(all_hidden_states)
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(all_hidden_states.dtype)
            loss = loss.to(all_hidden_states.dtype)

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=all_hidden_states,
        )


# LSTMForCausalLM.register_for_auto_class('AutoModelForCausalLM')
