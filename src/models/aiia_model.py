from torch.nn import Module
import torch
from torch_geometric.nn import to_hetero
from sklearn.decomposition import PCA
import joblib


class AIxIAModel(Module):
    def __init__(
        self,
        vit,
        pca: str | PCA,
        gnn,
        metadata,
        out_channels,
        hidden_dim=128,
        target_node: str = "style",
    ) -> None:
        super().__init__()
        self.pca = self.init_pca(pca)
        self.metadata = metadata
        self.target_node = target_node
        self.vit = vit
        self.gnn = to_hetero(gnn, self.metadata)
        self.head = torch.nn.Linear(
            in_features=(hidden_dim + (hidden_dim * out_channels)),
            out_features=out_channels,
        )

    def init_pca(self, pca):
        return pca if isinstance(pca, PCA) else joblib.load(pca)

    def forward(self, img, x_dict, edge_index_dict):
        img_feats = self.vit(img)
        device = img_feats.device
        img_feats = self.pca.transform(img_feats.detach().cpu().numpy())
        img_feats = torch.from_numpy(img_feats).to(device)
        kg_feats = self.gnn(x_dict, edge_index_dict)
        nodes = (
            kg_feats[self.target_node]
            .flatten()
            .unsqueeze(0)
            .repeat(img_feats.size(0), 1)
        )
        x = torch.cat([img_feats, nodes], axis=1).to(img_feats.device)
        return self.head(x)
