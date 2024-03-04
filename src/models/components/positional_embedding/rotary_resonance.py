import math
import torch
from einops import repeat

from src.models.components.positional_embedding.rotary_utils import find_correction_range, linear_ramp_mask, get_mscale


class ResonanceRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        assert dim % 2 == 0, 'dim must be multiple of 2 for Resonance RoPE.'

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._register_buffers(device)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.r_inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _register_buffers(self, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        r_wavelengths = torch.round(2 * math.pi / inv_freq)
        r_inv_freq = 2 * math.pi / r_wavelengths
        self.register_buffer("r_inv_freq", r_inv_freq, persistent=False)
        self.register_buffer("r_wavelengths", r_wavelengths, persistent=False)

    def compute_freqs(self, seq_len, device):
        freqs_list = list()
        for i in range(self.dim // 2):
            if seq_len >= self.r_wavelengths[i].item():
                t_i = torch.arange(self.r_wavelengths[i], device=device, dtype=self.r_inv_freq.dtype)
                current_freq = repeat(t_i * self.r_inv_freq[i], 'l -> (repeat l)',
                                      repeat=math.ceil(seq_len / self.r_wavelengths[i].item()))[:seq_len]
                freqs_list.append(current_freq)
            else:
                t_i = torch.arange(self.max_seq_len_cached, device=device, dtype=self.r_inv_freq.dtype)
                current_freq = t_i * self.r_inv_freq[i]
                freqs_list.append(current_freq)

        freqs = torch.stack(freqs_list, dim=1)
        return freqs

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > 70:
            print('Warning')

        freqs = self.compute_freqs(seq_len, device)
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


class ResonanceLinearScaledRotaryEmbedding(ResonanceRotaryEmbedding):
    """ResonanceRotaryEmbedding extended with linear scaling."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def compute_freqs(self, seq_len, device):
        freqs_list = list()
        for i in range(self.dim // 2):
            if seq_len < self.r_wavelength[i].item():
                t_i = torch.arange(self.r_wavelength[i], device=device, dtype=self.inv_freq.dtype)
                t_i /= self.scaling_factor
                current_freq = repeat(t_i * self.r_inv_freq[i], 'l -> (repeat l)',
                                      repeat=math.ceil(seq_len // self.r_wavelength[i].item()))[:seq_len]
                freqs_list.append(current_freq)
            else:
                t_i = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                t_i /= self.scaling_factor
                current_freq = t_i * self.r_inv_freq[i]
                freqs_list.append(current_freq)

        freqs = torch.stack(freqs_list, dim=1)
        return freqs


class ResonanceNTKScaledRotaryEmbedding(ResonanceRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        base = base * scaling_factor ** (dim / (dim - 2))
        super().__init__(dim, max_position_embeddings, base, device)


class ResonanceYaRNScaledRotaryEmbedding(ResonanceRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, attn_factor=1,
                 beta_fast=2, beta_slow=1):
        self.scale = scaling_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        super().__init__(dim, max_position_embeddings, base, device)

    def _register_buffers(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base,
                                          self.max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(
            device))  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        r_wavelengths = torch.round(2 * math.pi / inv_freq)
        r_inv_freq = 2 * math.pi / r_wavelengths
        self.register_buffer("r_inv_freq", r_inv_freq, persistent=False)
        self.register_buffer("r_wavelengths", r_wavelengths, persistent=False)

        self.mscale = float(
            get_mscale(self.scale) * self.attn_factor)  # Get n-d magnitude scaling corrected for interpolation

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        freqs = self.compute_freqs(seq_len, device)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)


if __name__ == '__main__':
    from src.models.components.positional_embedding import RotaryEmbedding

    emb_original = RotaryEmbedding(dim=32)
    emb = ResonanceRotaryEmbedding(dim=32)
    x = torch.tensor(list(range(64)), dtype=torch.float32)
    v_original = emb_original(x, seq_len=64)
    v_resonance = emb(x, seq_len=64)
    print(v_original)
    print(v_resonance)
