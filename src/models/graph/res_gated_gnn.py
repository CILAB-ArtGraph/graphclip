from typing import Any, Callable, Dict, Tuple
from torch_geometric.nn.conv import MessagePassing, ResGatedGraphConv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.resolver import activation_resolver


class ResGatedGNN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int | None = None,
        dropout: float = 0,
        act: str | Callable[..., Any] | None = "relu",
        act_first: bool = False,
        act_kwargs: Dict[str, Any] | None = None,
        norm: str | Callable[..., Any] | None = None,
        norm_kwargs: Dict[str, Any] | None = None,
        layer_act: str | Callable[..., Any] | None = "Sigmoid",
        layer_act_kwargs: Dict[str, Any] = {},
        jk: str | None = None,
        **kwargs
    ):
        self.layer_act = layer_act
        self.layer_act_kwargs = layer_act_kwargs
        super().__init__(
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout,
            act,
            act_first,
            act_kwargs,
            norm,
            norm_kwargs,
            jk,
            **kwargs
        )

    def init_conv(
        self, in_channels: int | Tuple[int, int], out_channels: int, **kwargs
    ) -> MessagePassing:
        layer_act = activation_resolver(self.layer_act, **self.layer_act_kwargs)
        kwargs.update({"act": layer_act})
        return ResGatedGraphConv(in_channels, out_channels, **kwargs)


if __name__ == "__main__":
    net = ResGatedGNN(1, 1, 1)
    print(net)
