from .clip_run import CLIPRun
from src.models import CLIPGraph
from open_clip import create_model_and_transforms
from open_clip import create_model
from copy import deepcopy
from .utils import ParameterKeys
from torch_geometric.nn.models.basic_gnn import BasicGNN
import torch_geometric.nn
from torch_geometric.data import HeteroData
import torch


class CLIPGraphRun(CLIPRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_general(self):
        super()._init_general()
        self.graph = self._init_graph()

    def _init_model(self) -> CLIPGraph:
        model_params = deepcopy(self.parameters).get(ParameterKeys.MODEL)
        clip_model = create_model(**model_params.get(ParameterKeys.VISUAL))
        gnn_model = self.__init_gnn()
        return CLIPGraph(
            clip_model=clip_model,
            gnn_model=gnn_model,
            metadata=self.graph.metadata(),
            **model_params.get(ParameterKeys.PARAMS, {})
        ).to(self.device)
    
    def __init_gnn(self) -> BasicGNN:
        gnn_params = deepcopy(self.parameters.get(ParameterKeys.MODEL).get(ParameterKeys.GNN))
        all_nets = torch_geometric.nn.__dict__
        net = all_nets[gnn_params.get(ParameterKeys.NAME)]
        return net(**gnn_params.get(ParameterKeys.PARAMS, {}))
    
    def _init_graph(self) -> HeteroData:
        graph_params = deepcopy(self.parameters.get(ParameterKeys.CLASS_SOURCE))
        return torch.load(**graph_params)
        
    
        