import torch
import open_clip
import torch_geometric as pyg
from torch_geometric.nn.models.basic_gnn import BasicGNN
from src.utils import StrEnum
import torch.nn.functional as F


class ReturnDict(StrEnum):
    IMAGE = "image_features"
    GRAPH = "graph_features"
    LOGIT_SCALE = "logit_scale"


class CLIPGraph(torch.nn.Module):
    def __init__(
        self,
        clip_model: open_clip.CLIP,
        gnn_model: BasicGNN,
        metadata: dict,
        target_node_t: str,
        use_logit_scale: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.visual = clip_model.visual
        hetero_gnn = pyg.nn.to_hetero(gnn_model, metadata)
        self.gnn = hetero_gnn
        self.target_node_t = target_node_t
        self.use_logit_scale = use_logit_scale
        self.logit_scale = clip_model.logit_scale
        assert target_node_t in metadata[0]

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_graph(self, x_dict, edge_index_dict, normalize: bool = False):
        graph = self.gnn(x_dict, edge_index_dict)
        target = graph[self.target_node_t]
        return F.normalize(target, dim=-1) if normalize else target

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
        # normalization
        image_features = (
            F.normalize(image_features, dim=1) if normalize else image_features
        )

        graph = self.gnn(x_dict, edge_index_dict)
        target = graph[self.target_node_t]
        graph_features = target[target_nodes]
        graph_features = (
            F.normalize(graph_features, dim=1) if normalize else graph_features
        )

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

    