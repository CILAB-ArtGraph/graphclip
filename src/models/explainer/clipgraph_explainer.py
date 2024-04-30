from torch import Tensor
from torch_geometric.data import Data, HeteroData
from .clip_explainer import VisualWrapperForExplanation
from .abstract_explainer import AbstractExplainer
from typing import Any, Union, Tuple, List
from torchvision.transforms import Compose
from open_clip import create_model_and_transforms
from src.models import CLIPGraph
from torch_geometric.explain import ExplainerAlgorithm, Explainer, HeteroExplanation
from .gradcam import apply_gradcam, visualize_gradcam, GradCAM
import torch
from copy import deepcopy
from yfiles_jupyter_graphs import GraphWidget


class GraphWrapperForExplanation(torch.nn.Module):
    def __init__(
        self, model: CLIPGraph, reference_feats: torch.Tensor, target_node: int
    ) -> None:
        super().__init__()
        self.backbone = model
        self.fake_head = torch.nn.Linear(
            in_features=reference_feats.size(1),
            out_features=reference_feats.size(0),
        )
        data_p = reference_feats.clone().detach().cpu()
        self.fake_head.weight = torch.nn.Parameter(data=data_p)
        self.target_node = target_node

    def forward(self, x_dict, edge_index_dict):
        nodes = self.backbone.encode_graph(
            x_dict=x_dict, edge_index_dict=edge_index_dict
        )[self.target_node].unsqueeze(dim=0)
        return self.fake_head(nodes)


class CLIPGraphExplainer(AbstractExplainer):
    def __init__(
        self, device: str = "cuda", image_preprocess: Union[str, Compose] = "clip"
    ) -> None:
        super().__init__()
        self.device = device
        self.image_preprocess = self._get_image_preprocess(
            image_preprocess=image_preprocess
        )

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

    def plot_explanation(
        self,
        explanation: HeteroExplanation,
        metadata: Tuple[List[str]],
        graph: HeteroData,
    ):
        subgraph_nodes, subgraph_edges = self.get_exp_subgraph(
            explanation=explanation, metadata=metadata
        )
        w = GraphWidget()

        # nodes
        w.nodes = [
            {"id": f"{t}_{idx}", "properties": {"label": t}}
            for t in subgraph_nodes.keys()
            for idx in subgraph_nodes[t]
        ]

        colors = [
            # yellow
            (255, 255, 0),
            # red
            (255, 0, 0),
            # green
            (0, 255, 0),
            # blue
            (0, 0, 255),
            # purple
            (255, 0, 255),
            # cyan
            (0, 255, 255),
            # orange
            (255, 165, 0),
            # pink
            (255, 192, 203),
            # brown
            (139, 69, 19),
            # grey
            (128, 128, 128),
            # black
            (0, 0, 0),
        ] * 3

        color_map = {node: color for node, color in zip(metadata[0], colors)}
        w.set_node_color_mapping(lambda x: color_map[x["properties"]["label"]])

        w.edges = [
            {"start": f"{h_t}_{h_idx}", "end": f"{t_t}_{t_idx}"}
            for (h_t, _, t_t) in subgraph_edges.keys()
            for (h_idx, t_idx) in subgraph_edges[(h_t, _, t_t)]
        ]

        return w

    def get_exp_subgraph(
        self,
        explanation: HeteroExplanation,
        metadata: Tuple[List[str]],
    ):
        exp_mod_nodes = {node_t: None for node_t in metadata[0]}
        exp_mod_edges = {edge_t: None for edge_t in metadata[1]}

        # nodes
        for node_t in metadata[0]:
            exp_mod_nodes[node_t] = list(
                map(
                    lambda x: x[0],
                    filter(
                        lambda x: x[1] > 0,
                        enumerate(explanation[node_t].node_mask.mean(dim=1).tolist()),
                    ),
                )
            )

        # edges
        for edge_t in metadata[1]:
            edges = list(
                map(
                    lambda x: x[0],
                    filter(
                        lambda x: x[1] > 0,
                        enumerate(explanation[edge_t].edge_mask.tolist()),
                    ),
                )
            )
            exp_mod_edges[edge_t] = explanation[edge_t].edge_index[:, edges].T.tolist()

        return exp_mod_nodes, exp_mod_edges

    def explain_graph(
        self,
        model: CLIPGraph,
        graph: Data | HeteroData,
        reference_feats: torch.Tensor,
        target_node: int,
        algo: ExplainerAlgorithm,
        algo_kwargs={},
        plot: bool = False,
    ):
        graph_model_wrap = GraphWrapperForExplanation(
            model=model,
            reference_feats=reference_feats,
            target_node=target_node,
        ).to(self.device)
        explainer = Explainer(model=graph_model_wrap, algorithm=algo, **algo_kwargs)
        explanation = explainer(graph.x_dict, edge_index=graph.edge_index_dict)
        return (
            explanation if not plot else self.plot_explanation(explanation=explanation)
        )
