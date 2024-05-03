from open_clip import CLIP
import torch
from copy import deepcopy


class ViT(torch.nn.Module):
    CLIP_HDIM = 512

    def __init__(
        self,
        model: CLIP,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = deepcopy(model.visual)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(
            in_features=ViT.CLIP_HDIM, out_features=out_channels
        )
        self.model = torch.nn.Sequential(self.backbone, self.dropout, self.linear)

    def forward(self, x):
        return self.model(x)
