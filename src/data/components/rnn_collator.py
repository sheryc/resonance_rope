from dataclasses import dataclass

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class DataCollatorForSeq2SeqRNNs(DataCollatorForSeq2Seq):
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        result_features = super().__call__(features, return_tensors)
        return result_features
