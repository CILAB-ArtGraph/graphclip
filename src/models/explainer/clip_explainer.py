from torch_geometric.data import Data, HeteroData
from .abstract_explainer import AbstractExplainer
import torch
from open_clip import CLIP


class VisualWrapperForExplanation(torch.nn.Module):
    def __init__(
        self, model: CLIP, text_reference_feat: torch.Tensor
    ) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=text_reference_feat.size(0), out_features=1
        )
        self.fake_head.weight = torch.nn.Parameter(
            data=text_reference_feat.copy().detach().cpu()
        )

    def forward(self, x):
        x = self.backbone.encode_image(x)
        return self.fake_head(x)
    
class TextualWrapperForExplanation(torch.nn.Module):
    def __init__(self, model: CLIP, image_reference_feat: torch.Tensor) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(in_features=image_reference_feat.size(0), out_features=1)
        self.fake_head.weight = torch.nn.Parameter(data=image_reference_feat.copy().detach().cpu())
    
    def forward(self, x):
        x = self.backbone.encode_text(x)
        return self.fake_head(x)


# TODO: update
class CLIPExplainer(AbstractExplainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def explain_image(self, img, text_reference):
        pass
    
    def explain_text(self, tokens: torch.Tensor, other: torch.Any):
        pass
    
    def explain_graph(self, graph: Data | HeteroData, other: torch.Any):
        raise NotImplementedError("Information modality not supported by CLIP!!")
