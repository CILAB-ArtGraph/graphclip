from .clip_graph import ReturnDict
import torch
from open_clip import CLIP
from torch_geometric.nn.models.basic_gnn import BasicGNN
import torch_geometric as pyg
import torch.nn.functional as F


class CLIPGraphMultiTask(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        gnn_model: BasicGNN,
        metadata: dict,
        target_nodes_t: list[str],
        use_logit_scale: bool = False,
    ) -> None:
        super().__init__()
        self.visual = clip_model.visual
        self.gnn = pyg.nn.to_hetero(gnn_model, metadata)
        self.target_nodes_t = target_nodes_t
        self.use_logit_scale = use_logit_scale
        self.logit_scale = clip_model.logit_scale
        assert all([target_node in metadata[0] for target_node in self.target_nodes_t])

    def encode_image(self, image, normalize: bool = False) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_graph(self, x_dict, edge_index_dict, normalize: bool = False) -> dict[str, torch.Tensor]:
        graph = self.gnn(x_dict, edge_index_dict)
        target = {node_t: graph[node_t] for node_t in self.target_nodes_t}
        return (
            {node_t: F.normalize(tensor, dim=-1) for node_t, tensor in target.items()}
            if normalize
            else target
        )

    def forward(
        self,
        image,
        x_dict,
        edge_index_dict,
        target_nodes: dict[str, list[int]],
        return_dict: bool = False,
        normalize: bool = False,
    ):
        image_features = self.encode_image(image=image, normalize=normalize)
        graph_features = self.encode_graph(x_dict=x_dict, edge_index_dict=edge_index_dict)
        graph_features = {node_t: tensor[target_nodes[node_t]] for node_t, tensor in graph_features.items()}

        if return_dict:
            out = {
                ReturnDict.IMAGE: image_features,
                ReturnDict.GRAPH: graph_features,
            }
            if self.use_logit_scale:
                out.update({ReturnDict.LOGIT_SCALE: self.logit_scale})
            return out

        if not self.use_logit_scale:
            return image_features, graph_features
        return image_features, graph_features, self.logit_scale
