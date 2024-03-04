from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import WhitespaceSplit


def get_num_tokenizer(modular_num: int):
    vocab = [str(i) for i in range(modular_num)] + ['[UNK]']
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocab)}, unk_token='[UNK]'))
    tokenizer.add_special_tokens(['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[EOS]'])
    tokenizer.add_tokens([str(i) for i in range(modular_num)])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.pad_token = '[PAD]'
    return tokenizer


if __name__ == '__main__':
    tokenizer = get_num_tokenizer(100)
    print(tokenizer.decode(tokenizer.encode('12 23 59 1 56').ids))
