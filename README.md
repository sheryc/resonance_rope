# Resonance RoPE and the PosGen Benchmark
<div align="center">
<a href="https://arxiv.org/abs/2403.00071"><img alt="arXiv Link" src="https://img.shields.io/badge/arXiv-2403.00071-blue"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description

This is the code for the paper "Resonance RoPE: Improving Context Length Generalization of Large Language Models". We provide the code for Resonance RoPE, Resonance YaRN and the PosGen benchmark.

## The Implementation of Resonance RoPE

The base class of Resonance RoPE:

```python
from einops import repeat
class ResonanceEmbedding(nn.Module):
    # The base class of the Resonance RoPE technique.
    def resonance_register(self):
        # This function rounds the wavelengths of all RoPE features to their closest integer based on self.inv_freq.
        r_wavelengths = torch.round(2 * math.pi / self.inv_freq)
        r_inv_freq = 2 * math.pi / r_wavelengths
        self.register_buffer("r_inv_freq", r_inv_freq, persistent=False)
        self.register_buffer("r_wavelengths", r_wavelengths, persistent=False)

    def compute_freqs(self, seq_len, device, dtype=None):
        # This function ensures that the pre-critical dimensions repeats the computed values.
        freqs_list = []
        dtype = self.r_inv_freq.dtype if not dtype else dtype
        for i in range(self.dim // 2):
            if seq_len >= self.r_wavelengths[i].item():
                t_i = torch.arange(self.r_wavelengths[i], device=device, dtype=dtype)
                current_freq = repeat(t_i * self.r_inv_freq[i].to(dtype), 'l -> (repeat l)',
                                      repeat=math.ceil(seq_len / self.r_wavelengths[i].item())).reshape(-1)[:seq_len]
            else:
                t_i = torch.arange(seq_len, device=device, dtype=dtype)
                current_freq = t_i * self.r_inv_freq[i].to(dtype)
            freqs_list.append(current_freq)
        freqs = torch.stack(freqs_list, dim=1)
        return freqs
```

When applying the Resonance RoPE technique to any existing RoPE scaling techniques, please make the following changes to the RoPE embedding class:

1. Create a new class inheriting both the original RoPE embedding and ``ResonanceEmbedding``.
2. After each computation or update of ``self.inv_freq``, rerun ``self.resonance_register()``.
3. Replace the following code snippet:
```python
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
```
to:
```python
    freqs = self.compute_freqs(self.max_seq_len_cached, device, self.r_inv_freq.dtype)
```

## Installation

```bash
# clone project
git clone https://github.com/sheryc/resonance_rope
cd resonance_rope

# [OPTIONAL] create conda environment
conda env create -f environment.yaml
conda activate resonance_rope

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

### The PosGen Benchmark Comparison

We provide the implementation of Resonance RoPE and Resonance YaRN in ``src/models/components/positional_embedding/rotary_resonance.py``.

To reproduce the comparison results on the PosGen benchmark used in the paper, please install hydra-ray-launcher first: https://github.com/facebookresearch/hydra/tree/main/plugins/hydra_ray_launcher.

To compare RoPE and Resonance RoPE / YaRN and Resonance YaRN, please run the following command:

```bash
python src/train.py -m multirun=ray data=alg_cot,alg_recurrent,alg_semirecurrent data.modular=17 experiment=sweep model.model.position_encoding_type=rotary_hf,rotary_scaled_yarn,rotary_resonance,rotary_resonance_yarn logger=aim model.compile=false trainer.precision=32 model.optimizer.lr=0.0002 trainer.min_epochs=150 trainer.max_epochs=150 seed=5549,4955,42,3701,49 mode.model.base=10000

```

If you want to run a single configuration, please use the format of the following command:

```bash
python src/train.py data=alg_semirecurrent data.modular=17 experiment=sweep model.model.position_encoding_type=rotary_resonance_yarn logger=aim model.compile=false trainer.precision=32 model.optimizer.lr=0.0002 trainer.min_epochs=150 trainer.max_epochs=150 seed=5549 mode.model.base=10000
```

### LLaMA Finetuning

We provide the code for LLaMA finetuning in ``llama_finetuning/``. This folder is a fork of [YaRN](https://github.com/jquesnelle/yarn), with Resonance YaRN added to the repo. Currently, we only tested Resonance YaRN's performance on LLaMA2-Chat models due to restrictions of hardware resources. To apply Resonance to any position encodings used in this repo, simply add ``--resonance-rope`` to your fine-tuning command. As an example:

```bash
cd llama_finetuning
```
```python
accelerate launch finetune.py \
    --output-dir output/resonance-yarn-7b-32k \
    --model meta-llama/Llama-2-7b-chat-hf \
    --scaling-factor 8 \
    --truncate 32768 \
    --max-train-steps 50 \
    --warmup-steps 2 \
    --architecture llama \
    --deepspeed \
    --resonance-rope
```

We provide this training script in ``llama_finetuning/train.sh``. For setting up the environments, please follow the instructions provided by the authors of YaRN in ``llama_finetuning/README.sh``.

## Citation

```
@misc{wang2024resonance,
    title={Resonance RoPE: Improving Context Length Generalization of Large Language Models},
    author={Suyuchen Wang and Ivan Kobyzev and Peng Lu and Mehdi Rezagholizadeh and Bang Liu},
    year={2024},
    eprint={2403.00071},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
