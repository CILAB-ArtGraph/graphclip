import torch
import open_clip
import torch_geometric as pyg
from torch_geometric.nn.models.basic_gnn import BasicGNN


class CLIPGraph(torch.nn.Module):
    def __init__(
        self,
        clip_model: open_clip.CLIP,
        gnn_model: BasicGNN,
        metadata: dict,
        target_node_t: str,
        **kwargs,
    ) -> None:
        super().__init__()
        self.visual = clip_model.visual
        hetero_gnn = pyg.nn.to_hetero(gnn_model, metadata)
        self.gnn = hetero_gnn
        self.target_node_t = target_node_t
        assert target_node_t in metadata[0]

    def forward(
        self,
        image,
        x_dict,
        edge_index_dict,
        target_nodes,
        return_dict: bool = False,
        normalize: bool = False,
    ):
        image_features = self.visual(image)
        graph = self.gnn(x_dict, edge_index_dict)
        target = graph[self.target_node_t]
        graph_features = target[target_nodes]

        return image_features, graph_features
