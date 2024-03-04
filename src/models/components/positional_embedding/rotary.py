import torch
import torch.nn as nn


class FixedRotaryPositionalEmbedding(nn.Module):
    def __init__(
            self, rotary_dim: int, rotary_base: int = 10000, max_position: int = 16384
    ):
        super().__init__()
        # This is an inverse frequency tensor
        # Each dimension has a higher denominator than the previous one
        # So, the frequency will be lower for higher dimensions
        inv_freq = 1.0 / (
                rotary_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )  # [rotary_dim/2]

        # Now, we create frequencies for each position
        t = torch.arange(max_position, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_position, rotary_dim/2]

        sins = torch.sin(freqs)
        coss = torch.cos(freqs)

        emb = torch.cat([sins, coss], dim=-1)  # [max_position, rotary_dim]
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    """
    Example: [a, b, c, d] -> [-b, a, -d, c]
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset: x.shape[1] + offset, None, :].repeat_interleave(2, 3),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)
