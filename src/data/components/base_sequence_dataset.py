import os
from itertools import combinations, cycle
from typing import List

from datasets import load_dataset
from math import sin, pow, floor, pi
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.models.components.tokenizers.number_tokenizer import get_num_tokenizer
from src.utils import create_directories_for_path, RankedLogger


class SequenceDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 modular: int,
                 seq_len: int,
                 train_size: int,
                 val_size: int,
                 test_size: int,
                 near_count: int = 2,
                 far_count: int = 1,
                 current_split: str = 'train',
                 mask_initial_tokens: bool = True,
                 arith_task: str = 'mod_addition',
                 model_mode='transformers'):
        super().__init__()
        assert seq_len > near_count + far_count, 'Minimum length should be larger than near_count + far_count'
        assert model_mode in ['transformers', 'rnns'], 'Unknown model mode'
        self.modular = modular
        padding_side = 'left' if model_mode == 'transformers' and current_split == 'train' else 'right'
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=get_num_tokenizer(modular),
                                                 padding_side=padding_side,
                                                 pad_token='[PAD]')
        self.seq_len = seq_len
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        total_data = train_size + val_size + test_size
        self.near_count = near_count
        self.far_count = far_count
        self.total_dependency_count = near_count + far_count
        self.current_split = current_split
        self.mask_initial_tokens = mask_initial_tokens
        self.arith_task = arith_task
        self.model_mode = model_mode
        self.filename = os.path.join(data_dir, self.arith_task, self.dataset_name(),
                                     f'mod{modular}_n{near_count}f{far_count}',
                                     f'len{seq_len}-{current_split}.txt')
        self.complete_data_filename = os.path.join(data_dir, self.arith_task, self.dataset_name(),
                                                   f'mod{modular}_n{near_count}f{far_count}',
                                                   f'data_total{total_data}.txt')
        create_directories_for_path(self.filename)
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def atom_arithmetic(self, far: List[int], near: List[int]):
        if self.arith_task == 'mod_addition':
            return int((sum(far) + sum(near)) % self.modular)
        elif self.arith_task == 'pow_sin':
            return floor(pow(sin(sum(far) + sum(near)), 2) * self.modular)
        elif self.arith_task == 'pow_sin_2pi':
            s = (sum(far) + sum(near)) % self.modular
            return floor(abs(sin(2 * pi * s / self.modular)) * self.modular)
        elif self.arith_task == 'str_hash':
            result = hash(''.join([str(i) for i in near + far]))
            return result % self.modular

    def load_data(self):
        if self.current_split == 'val':
            sp = 'validation'
        else:
            sp = self.current_split
        data = load_dataset('text', split=sp, data_files={sp: self.filename})
        self.data = data

    def _generate_complete_data(self):
        total_data = self.train_size + self.val_size + self.test_size
        data = []
        i = 0
        for seq in cycle(combinations(range(self.modular), self.total_dependency_count)):
            length = 512
            sequence = list(seq)
            sequence = self._extend_sequence(sequence, length)
            sequence = [str(num) for num in sequence]
            sequence = ' '.join(sequence)
            data.append(sequence)
            i += 1
            if i == total_data:
                break
        with open(self.complete_data_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))

    def generate_data(self):
        self.get_logger().info(f'Generating sequence dataset for the {self.current_split} split...')
        if not os.path.exists(self.complete_data_filename):
            self._generate_complete_data()
        with open(self.complete_data_filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
        if self.current_split == 'train':
            data = [' '.join(line.strip().split()[:self.seq_len]) for line in data[:self.train_size]]
        elif self.current_split == 'val':
            # data = [' '.join(line.strip().split()[:self.seq_len_max]) for line in
            #         data[self.train_size:self.train_size + self.val_size]]
            data = [' '.join(line.strip().split()[:self.seq_len]) for line in
                    data[:self.val_size]]
        else:
            # data = [' '.join(line.strip().split()[:self.seq_len_max]) for line in data[-self.test_size:]]
            data = [' '.join(line.strip().split()[:self.seq_len]) for line in
                    data[self.val_size:self.val_size + self.test_size]]
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(data))

    def _extend_sequence(self, sequence, length):
        raise NotImplementedError

    def process_instance(self, example):
        total_input_count = self.near_count + self.far_count + 1
        instance_ids = self.tokenizer('[EOS] ' + example['text'], truncation=False, add_special_tokens=False,
                                      max_length=self.seq_len).input_ids

        if self.current_split != 'train':
            input_ids = instance_ids[:total_input_count]
        else:
            input_ids = instance_ids

        attention_mask = [1] * len(input_ids)

        if self.mask_initial_tokens:
            labels = [-100] * total_input_count + instance_ids[total_input_count:]
        else:
            labels = instance_ids

        if self.model_mode == 'transformers':
            result = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        else:
            result = {
                'input_ids': input_ids,
                'labels': labels,
                'lengths': len(labels)
            }
        return result

    @staticmethod
    def get_logger():
        return RankedLogger(__name__, rank_zero_only=True)

    @classmethod
    def dataset_name(cls) -> str:
        return ''
