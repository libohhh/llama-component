import torch

def precompute_theta_pos_frequencies(dim: int, seq_len: int, theta: float = 10000.0):
    assert dim % 2 == 0, "Dimension must be divisible by 2"
    # theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Dim / 2)
    theta_numerator = torch.arange(0, dim, 2).float()
    # Shape: (Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / dim))# (Dim / 2)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len)
    # Shape: (Seq_Len) outer_product* (Dim / 2) -> (Seq_Len, Dim / 2)
    freqs = torch.outer(m, theta).float()
    # complex numbers c = R * exp(m * theta), R = 1 :
    # (Seq_Len, Dim / 2) -> (Seq_Len, Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # (B, Seq_Len, H, Dim) -> (B, Seq_Len, H, Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Dim/2) --> (1, Seq_Len, 1, Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_Len, H, Dim/2) * (1, Seq_Len, 1, Dim/2) = (B, Seq_Len, H, Dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, Seq_Len, H, Dim/2) -> (B, Seq_Len, H, Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Dim/2, 2) -> (B, Seq_Len, H, Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)