import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Initialize learnable parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        variance = torch.var(x, dim=self.dim, unbiased=False, keepdim=True)

        # Normalize input tensor
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        # Apply gamma and beta
        output = self.gamma * x_normalized + self.beta

        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize learnable gamma parameters
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        x_normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = self.weight * x_normalized
        return output


class BatchNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Initialize learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):
        
        batch_mean = torch.mean(x, dim=0, keepdim=False)
        batch_var = torch.var(x, dim=0, unbiased=False, keepdim=False)

        # Normalize input
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        output = self.gamma * x_normalized + self.beta

        return output
