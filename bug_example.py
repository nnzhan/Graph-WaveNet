import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.vector = nn.Parameter(torch.randn(5, 5).to(device), requires_grad=True)

model_cpu = Model('cpu')
assert 'vector' in model_cpu.state_dict()
model_gpu = Model('cuda:0')
assert 'vector' in model_gpu.state_dict() # Raise assertionerror

