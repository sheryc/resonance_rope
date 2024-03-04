import math
import torch

from src.models.components.positional_embedding.rotary_utils import find_correction_range, linear_ramp_mask, get_mscale


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=torch.get_default_dtype())

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class LinearScaledRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class NTKScaledRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__(dim, max_position_embeddings, base, device)
        base = base * scaling_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # inv_freq = math.pi * inv_freq
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            # emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            emb = freqs
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(torch.get_default_dtype()),
                                 persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(torch.get_default_dtype()),
                                 persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class YaRNScaledRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, attn_factor=1,
                 beta_fast=2, beta_slow=1):
        super().__init__(dim, max_position_embeddings, base, device)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scaling_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            # emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            emb = freqs

            self.register_buffer("cos_cached",
                                 (emb.cos() * self.mscale)[None, None, :, :].to(torch.get_default_dtype()),
                                 persistent=False)
            self.register_buffer("sin_cached",
                                 (emb.sin() * self.mscale)[None, None, :, :].to(torch.get_default_dtype()),
                                 persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base,
                                          self.max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(
            device))  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(
            get_mscale(self.scale) * self.attn_factor)  # Get n-d magnitude scaling corrected for interpolation


if __name__ == '__main__':
    emb = NTKScaledRotaryEmbedding(dim=32, max_position_embeddings=64, base=(10 * math.pi), device=None,
                                   scaling_factor=1.0)
    x = torch.tensor(list(range(64)))
    print(emb(x, seq_len=64))
