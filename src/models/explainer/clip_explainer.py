from torch_geometric.data import Data, HeteroData
from .abstract_explainer import AbstractExplainer
import torch
from open_clip import CLIP
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from einops import rearrange

class VisualWrapperForExplanation(torch.nn.Module):
    def __init__(self, model: CLIP, text_reference_feat: torch.Tensor) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=text_reference_feat.size(1), out_features=text_reference_feat.size(0)
        )
        text_p = text_reference_feat.clone().detach().cpu()
        self.fake_head.weight = torch.nn.Parameter(
            data=text_p,
        )

    def forward(self, x):
        x = self.backbone.encode_image(x)
        return self.fake_head(x)


class TextualWrapperForExplanation(torch.nn.Module):
    def __init__(self, model: CLIP, image_reference_feat: torch.Tensor) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=image_reference_feat.size(0), out_features=1
        )
        self.fake_head.weight = torch.nn.Parameter(
            data=image_reference_feat.clone().detach().cpu()
        )

    def forward(self, x):
        x = self.backbone.encode_text(x)
        return self.fake_head(x)


# TODO: update
class CLIPExplainer(AbstractExplainer):
    "Per le immagini serve un'implementazione dignitosa di GRAD-Cam"
    "Per il testo anche"
    def __init__(self, visual_explainer_cls, text_explainer_cls) -> None:
        super().__init__()
        self.visual_explainer_cls = visual_explainer_cls
        self.text_explainer_cls = text_explainer_cls

    def explain_image(self, backbone: CLIP, target_layers: list[str], img: torch.Tensor, text_reference: torch.Tensor, target: int, **kwargs):
        visual_wrapper = VisualWrapperForExplanation(model=backbone, text_reference_feat=text_reference).to('cuda')
        target_layers = [visual_wrapper.backbone.visual.transformer.resblocks[-1].ln_1]
        # target_layers = [getattr(visual_wrapper, x) for x in target_layers]
        visual_exp = self.visual_explainer_cls(visual_wrapper, target_layers, **kwargs)
        targets = None if not target else [ClassifierOutputTarget(target)]
        return visual_exp(img, targets)

    def explain_text(self, tokens: torch.Tensor, other: torch.Any):
        pass

    def explain_graph(self, graph: Data | HeteroData, other: torch.Any):
        raise NotImplementedError("Information modality not supported by CLIP!!")
