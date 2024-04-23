from typing import Tuple
from torch_geometric.nn.conv import MessagePassing, TransformerConv
from torch_geometric.nn.models.basic_gnn import BasicGNN


class TransformerGNN(BasicGNN):
    def init_conv(self, in_channels: int | Tuple[int], out_channels: int, **kwargs) -> MessagePassing:
        return TransformerConv(in_channels, out_channels, **kwargs)