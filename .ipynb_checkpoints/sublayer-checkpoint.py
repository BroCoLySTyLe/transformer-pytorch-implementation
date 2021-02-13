import torch.nn as nn
import torch
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, eps=1e-6):
        super(SublayerConnection, self).__init__()
        
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        LayerNorm = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
        return x + self.dropout(sublayer(LayerNorm))
    

