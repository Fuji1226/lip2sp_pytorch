from turtle import forward
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return input_forward
 
    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None


class GradientReversal(nn.Module):
    def __init__(self, scale: float):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
            nn.Conv1d(1, 1, kernel_size=1),
            nn.Conv1d(1, 1, kernel_size=1),
        )
        self.fc = nn.Conv1d(1, 1, kernel_size=1)
        self.grl = GradientReversal(scale=1.0)

    def forward(self, x):
        out = self.layers(x)
        out = self.grl(out)
        out = self.fc(out)
        return out