from transformers import AutoConfig, PretrainedConfig


def get_hf_config(**kwargs) -> PretrainedConfig:
    pretrained_name = kwargs.pop("hf_model_name", None)
    base_pretrained_config_name = kwargs.pop("hf_template_config", "t5-base")
    if pretrained_name is None:
        config = AutoConfig.from_pretrained(base_pretrained_config_name)
        config.update(kwargs)
        return config
    else:
        return AutoConfig.from_pretrained(pretrained_name)


class LSTMConfig(PretrainedConfig):
    # From https://huggingface.co/grostaco/lstm-base/tree/main
    model_type = 'LSTM'

    def __init__(self, vocab_size: int = 30522, embedding_dim: int = 300, hidden_dim: int = 300, n_layers: int = 2,
                 bidirectional: bool = False, classifier_dropout: float = .5, pad_index: int = 0, **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.classifier_dropout = classifier_dropout
        self.pad_index = pad_index
        super().__init__(**kwargs)


if __name__ == '__main__':
    print(get_hf_config(hf_template_config='t5-base'))
