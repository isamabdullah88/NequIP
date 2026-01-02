
import torch.nn as nn
import e3nn.math as emath
from e3nn import o3
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter


def ploynomial_cutoff(x, rcut):

    p = 6
    envelope = 1 - (p * (x / rcut)**5) + ((p - 1) * (x / rcut)**6)

    mask = (x < rcut).to(x.dtype)
    return envelope * mask


class Radial(nn.Module):
    def __init__(self, in_dim, out_dim, rcut):
        super(Radial, self).__init__()
        self.rcut = rcut
        self.numbasis = in_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )

    def forward(self, dist):
        bessel = emath.soft_one_hot_linspace(dist, 0.0, 5.0, self.numbasis, basis='bessel', cutoff=True)
        # print("Bessel basis shape:", bessel.shape)
        distf = self.model(bessel)

        cutoff = ploynomial_cutoff(dist, self.rcut).unsqueeze(-1)

        distf = distf * cutoff

        return distf

        
class Convolution(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numbasis=10, rcut=5.0):
        super(Convolution, self).__init__()

        in_irreps = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        # print('in irreps:', in_irreps)
        self.avg_neighbors = 25.0
        
        self.numbasis = numbasis
        self.rcut = rcut

        self.tp = o3.FullyConnectedTensorProduct(
            o3.Irreps(in_irreps),  # input features
            o3.Irreps("1x0e + 1x1o + 1x2e"),   # spherical harmonics l=1
            o3.Irreps(in_irreps),  # output features
            shared_weights=False,
            internal_weights=False
        )

        numweights = self.tp.weight_numel
        # print("Number of weights in TP:", numweights)
        self.radialMLP = Radial(self.numbasis, numweights, self.rcut)

        # self.linear = o3.Linear(o3.Irreps(in_irreps), o3.Irreps(in_irreps))

    def forward(self, nodes, pos, batch):

        edgeidxs = radius_graph(pos, r=self.rcut, batch=batch, max_num_neighbors=100)

        src, dst = edgeidxs

        # print('nodes shape:', nodes.shape)
        neighbors = nodes[src]

        posvec = pos[src] - pos[dst]

        dist = posvec.norm(dim=1, keepdim=False)

        radial = self.radialMLP(dist)

        ylm = o3.spherical_harmonics(l=[0, 1, 2], x=posvec, normalize=True, normalization='component')

        # print('neighbors shape:', neighbors.shape)
        # print("Ylm shape:", ylm.shape)
        # print("Radial shape:", radial.shape)

        messages = self.tp(neighbors, ylm, weight=radial)

        aggregated = scatter(messages, dst, dim=0, reduce='add')

        aggregated = aggregated / (self.avg_neighbors**0.5)

        return aggregated