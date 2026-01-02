import torch
import torch.nn as nn
from e3nn import o3
from .convolution import Convolution
from .gate import NonLinearGate

class InteractionBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim):
        super(InteractionBlock, self).__init__()

        # self-interaction
        in_irreps = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2o")
        self.owninteraction = o3.Linear(in_irreps, in_irreps)

        # convolution
        self.convolution = Convolution(l0dim, l1dim, l2dim, mps=False)

        # gate
        self.gate = NonLinearGate(l0dim, l1dim, l2dim)

    def forward(self, nodes: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        self_interacted1 = self.owninteraction(nodes)

        # print("Self-interacted features shape:", self_interacted1.shape)

        convolved = self.convolution(self_interacted1, pos, batch)

        self_interacted2 = self.owninteraction(convolved)

        mixed = nodes + self_interacted2

        gated = self.gate(mixed)

        return gated