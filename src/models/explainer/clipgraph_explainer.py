from torch import Tensor
from torch_geometric.data import Data, HeteroData
from .clip_explainer import VisualWrapperForExplanation
from .abstract_explainer import AbstractExplainer
from typing import Any, Union
from torchvision.transforms import Compose
from open_clip import create_model_and_transforms
from src.models import CLIPGraph
from torch_geometric.explain import ExplainerAlgorithm, Explainer
from .gradcam import apply_gradcam, visualize_gradcam, GradCAM
import torch


class GraphWrapperForExplanation(torch.nn.Module):
    def __init__(self, model: CLIPGraph, reference_feats: torch.Tensor) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=reference_feats.size(1),
            out_features=reference_feats.size(0),
        )
        data_p = reference_feats.clone().detach().cpu()
        self.fake_head.weight = torch.nn.Parameter(data=data_p)

    def forward(self, x_dict, edge_index_dict):
        nodes = self.backbone.encode_graph(
            x_dict=x_dict, edge_index_dict=edge_index_dict
        )
        return self.fake_head(nodes)


class CLIPGraphExplainer(AbstractExplainer):
    def __init__(
        self, device: str = "cuda", image_preprocess: Union[str, Compose] = "clip"
    ) -> None:
        super().__init__()
        self.device = device
        self.image_preprocess = self._get_image_preprocess

    def _get_image_preprocess(self, image_preprocess: Union[str, Compose]) -> Compose:
        if isinstance(image_preprocess, Compose):
            return image_preprocess
        return create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[
            2
        ]

    def explain_image(
        self,
        img_path: str,
        model: CLIPGraph,
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

    def explain_graph(
        self,
        model: CLIPGraph,
        graph: Data | HeteroData,
        reference_feats: torch.Tensor,
        algo: ExplainerAlgorithm,
        algo_kwargs={},
        plot: bool = False,
    ):
        graph_model_wrap = GraphWrapperForExplanation(
            model=model, reference_feats=reference_feats
        ).to(self.device)
        explainer = Explainer(model=graph_model_wrap, algorithm=algo, **algo_kwargs)
        explanation = explainer(graph.x_dict, edge_index=graph.edge_index_dict)
        return explanation if not plot else explanation.visualize_graph()
