from typing import Tuple
from torch_geometric.nn.conv import MessagePassing, ResGatedGraphConv
from torch_geometric.nn.models.basic_gnn import BasicGNN


class ResGatedGNN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(
        self, in_channels: int | Tuple[int, int], out_channels: int, **kwargs
    ) -> MessagePassing:
        return ResGatedGraphConv(in_channels, out_channels, **kwargs)


if __name__ == "__main__":
    net = ResGatedGNN(1, 1, 1)
    print(net)
