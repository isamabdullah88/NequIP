import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from gaussian import gaussian_expand_torch
from .embedding import AtomEmbedding
from .interaction import InteractionBlock   
from .output import OutputBlock


def force(energy, pos):
    ones = torch.ones_like(energy)

    grads = torch.autograd.grad(
        outputs=energy,
        inputs=pos,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    forces = -grads
    return forces


class NequIP(nn.Module):
    def __init__(self):
        super(NequIP, self).__init__()
        
        self.l0dim = 16
        self.l1dim: int = 8
        self.l2dim: int = 4

        self.atomembeds = AtomEmbedding(self.l0dim, self.l1dim, self.l2dim)

        self.interaction_block = InteractionBlock(self.l0dim, self.l1dim, self.l2dim)

        self.output_block = OutputBlock(self.l0dim, self.l1dim, self.l2dim)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        nodes = self.atomembeds(z)

        interacted1 = self.interaction_block(nodes, pos, batch)

        interacted2 = self.interaction_block(interacted1, pos, batch)

        output = self.output_block(interacted2, z)
        # print('output shape:', output.shape)

        energyt = global_add_pool(output, batch)
        # print('energy shape:', energyt.shape)

        return energyt
    

