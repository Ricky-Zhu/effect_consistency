import torch

a = torch.rand([128, 1], dtype=torch.float32)
b = torch.rand([128, 1], dtype=torch.float32)
h = torch.rand([128, 1], dtype=torch.float32)
c = 0.95
d = h + c * a * b
