import math

import torch
from einops import repeat


# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaYaRNScaledResonanceRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048,
                 extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        freqs = self.compute_freqs(max_position_embeddings, device=device)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

    def resonance_register(self, inv_freq):
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

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            # Resonance repetition
            freqs = self.compute_freqs(seq_len, device=x.device)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            # emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            emb = freqs
            self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype),
                                 persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype),
                                 persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base,
                                          self.original_max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(
            device)) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        # Adding resonance
        self.resonance_register(inv_freq)
        self.mscale = float(
            get_mscale(self.scale) * self.attn_factor)  # Get n-d magnitude scaling corrected for interpolation
