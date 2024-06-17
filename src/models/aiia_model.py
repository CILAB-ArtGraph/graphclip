from torch.nn import Module
import torch
from torch_geometric.nn import to_hetero


class AIxIAModel(Module):
    def __init__(self, vit, gnn, metadata, out_channels, hidden_dim=128, target_node: str = "style") -> None:
        super().__init__()
        self.metadata = metadata
        self.target_node = target_node
        self.vit = vit
        self.gnn = to_hetero(gnn, self.metadata)
        self.head = torch.nn.Linear(
            in_features=(hidden_dim + (hidden_dim * out_channels)), out_features=out_channels
        )

    def forward(self, img, x_dict, edge_index_dict):
        img_feats = self.vit(img)
        kg_feats = self.gnn(x_dict, edge_index_dict)
        nodes = kg_feats[self.target_node].flatten()
        x = torch.cat([img_feats, nodes]).to(img_feats.device)
        return self.head(x)