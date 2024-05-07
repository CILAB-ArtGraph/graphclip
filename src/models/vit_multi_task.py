from open_clip import CLIP
import torch
from typing import Optional
from copy import deepcopy


class ViTMultiTask(torch.nn.Module):
    CLIP_HDIM = 512

    def __init__(
        self,
        model: CLIP,
        out_channels: list[int],
        tasks: str,
        dropout: float = 0.0,
        output_dict: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.backbone = deepcopy(model.visual)
        self.out_channels = out_channels
        self.tasks = tasks
        self.dropout = dropout
        self.output_dict = output_dict
        self.init()

    def init(self):
        self.modules = torch.nn.ModuleDict(
            {
                task: torch.nn.Sequential(
                    torch.nn.Dropout(p=self.dropout),
                    torch.nn.Linear(
                        in_features=ViTMultiTask.CLIP_HDIM,
                        out_features=self.out_channels[idx],
                    ),
                )
                for (idx, task) in enumerate(self.tasks)
            }
        )

    def forward(self, x):
        features = self.backbone(x)
        return (
            {task: self.modules[task](features) for task in self.tasks}
            if self.output_dict
            else [self.modules[task](features) for task in self.tasks]
        )
