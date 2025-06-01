import torch
import math
import matplotlib.pyplot as plt

def get_positional_encoding(max_len=50, d_model=128):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def get_pe_2(max_len=50, d_model=128):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len)
    pos = pos.unsqueeze(dim=1)
    _2i = torch.arange(0, d_model, step=2)
    pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    return pe

pe = get_positional_encoding()

plt.imshow(pe, cmap='coolwarm_r', aspect='auto')
plt.colorbar()
plt.title('Full Positional Encoding Matrix')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.show()
