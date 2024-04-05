import torch
from typing import Any, Union
from torch_geometric.data import Data, HeteroData


class AbstractExplainer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def explain_image(self, image: torch.Tensor, other: Any):
        raise NotImplementedError()
    
    def explain_text(self, tokens: torch.Tensor, other: Any):
        raise NotImplementedError()
    
    def explain_graph(self, graph: Union[Data, HeteroData], other: Any):
        raise NotImplementedError()