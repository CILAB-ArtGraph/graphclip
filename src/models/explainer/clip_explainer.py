from torch_geometric.data import Data, HeteroData
from .abstract_explainer import AbstractExplainer
import torch
from open_clip import CLIP, SimpleTokenizer, get_tokenizer, create_model_and_transforms
from torch import nn
from torch.nn import functional as F
from .gradcam import apply_gradcam, visualize_gradcam, GradCAM
from typing import Union
from torchvision.transforms import Compose
from .textgrad import interpret_sentence


def text_global_pool(x, text: torch.Tensor = None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class VisualWrapperForExplanation(torch.nn.Module):
    def __init__(self, model: CLIP, reference_feat: torch.Tensor) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=reference_feat.size(1),
            out_features=reference_feat.size(0),
        )
        text_p = reference_feat.clone().detach().cpu()
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


class CLIPExplainer(AbstractExplainer):
    def __init__(
        self,
        device: str = "cuda",
        image_preprocess: Union[str, Compose] = "clip",
        tokenizer: Union[str, SimpleTokenizer] = "clip",
    ) -> None:
        super().__init__()
        self.device = device
        self.image_preprocess = self._get_image_preprocess(
            image_preprocess=image_preprocess
        )
        self.tokenizer = self._get_tokenizer(tokenizer=tokenizer)

    def _get_image_preprocess(self, image_preprocess: Union[str, Compose]) -> Compose:
        if isinstance(image_preprocess, Compose):
            return image_preprocess
        return create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[
            2
        ]

    def _get_tokenizer(self, tokenizer: Union[str, SimpleTokenizer]) -> SimpleTokenizer:
        if isinstance(tokenizer, SimpleTokenizer):
            return tokenizer
        return get_tokenizer("ViT-B-32")

    def explain_image(
        self,
        img_path: str,
        model: CLIP,
        reference_feats: torch.Tensor,
        target: int,
        overlayed: bool = True,
    ):
        # apply gradcam to the image
        vis_wrap_model = VisualWrapperForExplanation(
            model=model, reference_feat=reference_feats
        ).to(self.device)
        gradcam = GradCAM(model=vis_wrap_model)
        cam = apply_gradcam(
            image_path=img_path,
            preprocess=self.image_preprocess,
            model=vis_wrap_model,
            gradcam=gradcam,
            target=target,
            device=self.device,
        )
        return (
            cam
            if not overlayed
            else visualize_gradcam(image_path=img_path, gweights=cam)
        )

    def explain_text(
        self,
        model: CLIP,
        text: str,
        image_reference_feats: torch.Tensor,
        plot: bool = False,
    ):
        text_wrap_model = TextualWrapperForExplanation(
            model=model, image_reference_feat=image_reference_feats
        )
        return interpret_sentence(
            model=text_wrap_model,
            sentence=text,
            tokenizer=self.tokenizer,
            device=self.device,
            plot=plot,
        )

    def explain_graph(self, graph: Data | HeteroData, other: torch.Any):
        raise NotImplementedError("Information modality not supported by CLIP!!")
