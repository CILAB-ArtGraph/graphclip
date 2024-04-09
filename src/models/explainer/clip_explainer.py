from torch_geometric.data import Data, HeteroData
from .abstract_explainer import AbstractExplainer
import torch
from open_clip import CLIP
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def text_global_pool(x, text: torch.Tensor = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


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
        
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.text_projection = model.text_projection
        self.ln_final = model.ln_final
        self.attn_mask = model.attn_mask
        self.text_pool_type = model.text_pool_type
        
        self.fake_head = torch.nn.Linear(
            in_features=image_reference_feat.size(0), out_features=1
        )
        self.fake_head.weight = torch.nn.Parameter(
            data=image_reference_feat.clone().detach().cpu()
        )
        
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, x):
        x = self.encode_text(x)
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
