from torch.nn import Module
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn import to_hetero
from sklearn.decomposition import PCA
import joblib
import torch


class AIxIAMultiTaskModel(Module):
    def __init__(
        self,
        vit: Module,
        pca: PCA | str,
        gnn: BasicGNN,
        metadata: dict,
        target_nodes_t: list[str],
        out_channels: dict[str, int],
        num_layers: int = 1,
        hidden_dim: int = 128,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        self.pca = self.init_pca(pca)
        self.metadata = metadata
        self.return_dict = return_dict
        self.target_nodes_t = target_nodes_t
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.vit = vit
        self.gnn = to_hetero(gnn, self.metadata)
        self.seq = self.init_sequential()
        self.heads = torch.nn.ModuleDict(
            {
                task: torch.nn.Linear(
                    in_features=self.seq[-2].out_features,
                    out_features=out_channels[task],
                )
                for task in self.target_nodes_t
            }
        )

    def init_pca(self, pca):
        return pca if isinstance(pca, PCA) else joblib.load(pca)

    def init_sequential(self):
        in_feats = self.hidden_dim + sum(
            map(
                lambda x: self.out_channels[x] * self.hidden_dim,
                self.target_nodes_t,
            )
        )
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                torch.nn.Linear(in_features=in_feats, out_features=in_feats // 2)
            )
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            in_feats = in_feats // 2
        return torch.nn.Sequential(*layers)

    def encode_kg(self, x_dict, edge_index_dict):
        kg_feats = self.gnn(x_dict, edge_index_dict)
        kg_feats = {k: v for k, v in kg_feats.items() if k in self.target_nodes_t}
        nodes = [v.flatten() for v in kg_feats.values()]
        return torch.cat(nodes)

    def forward(self, img, x_dict, edge_index_dict):
        img_feats = self.vit(img)
        device = img_feats.device
        img_feats = self.pca.transform(img_feats.detach().cpu().numpy())
        img_feats = torch.from_numpy(img_feats).to(device)
        kg_feats = self.encode_kg(x_dict, edge_index_dict)
        node_feats = kg_feats.unsqueeze(0).repeat(img_feats.size(0), 1)
        x = torch.cat([img_feats, node_feats], axis=1).to(img_feats.device)
        x = self.seq(x)
        return (
            {k: head(x) for k, head in self.heads.items()}
            if self.return_dict
            else [head(x) for k, head in self.heads.items()]
        )
