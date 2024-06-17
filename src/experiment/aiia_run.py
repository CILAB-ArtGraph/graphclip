from src.models import AIxIAModel
from .clip_run import CLIPRun
from copy import deepcopy
import torch
from .utils import ParameterKeys
from src.data import DataDict
from timm import create_model
from torch_geometric.nn.models.basic_gnn import BasicGNN
import src.models.graph as graph_module
from torch_geometric.data import HeteroData


class AIxIARun(CLIPRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_model(self) -> AIxIAModel:
        model_params = deepcopy(self.parameters).get(ParameterKeys.MODEL)
        vit = create_model(**model_params.get(ParameterKeys.VISUAL))
        vit.reset_classifier(model_params.get(ParameterKeys.PARAMS).get("hidden_dim"))
        gnn_model = self.__init_gnn()
        return AIxIAModel(
            vit=vit,
            gnn=gnn_model,
            metadata=self.graph.metadata(),
            **model_params.get(ParameterKeys.PARAMS, {}),
        )

    def __init_gnn(self) -> BasicGNN:
        gnn_params = deepcopy(
            self.parameters.get(ParameterKeys.MODEL).get(ParameterKeys.GNN)
        )
        all_nets = graph_module.__dict__
        net = all_nets[gnn_params.get(ParameterKeys.NAME)]
        return net(**gnn_params.get(ParameterKeys.PARAMS, {}))

    def _init_graph(self) -> HeteroData:
        graph_params = deepcopy(self.parameters.get(ParameterKeys.CLASS_SOURCE))
        return torch.load(**graph_params)

    def _init_general(self):
        super()._init_general()
        self.graph = self._init_graph()

    def train_epoch(self, epoch):
        cumulated_loss = 0.0
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAINING} at epoch {epoch}/{self.num_epochs}",
        )

        self.model.train()
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            gts = data_dict[DataDict.GTS].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images, self.graph.x_dict, self.graph.edge_index_dict)
            loss_out = self.criterion(out, gts)
            loss_out.backward()
            self.optimizer.step()
            batch_loss = loss_out.cpu().item()

            cumulated_loss = cumulated_loss + batch_loss

            self.schedule(phase=ParameterKeys.TRAINING)

        cumulated_loss = cumulated_loss / len(self.train_loader)
        self.print_stats(
            epoch=epoch, cumulated_loss=cumulated_loss, phase=ParameterKeys.TRAINING
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        cumulated_loss = 0.0
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VALIDATION} at epoch {epoch}/{self.num_epochs}",
        )

        self.model.eval()
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            gts = data_dict[DataDict.GTS].to(self.device)

            out = self.model(image=images)
            loss_out = self.criterion(out, gts)
            batch_loss = loss_out.cpu().item()

            # update stats
            cumulated_loss = cumulated_loss + batch_loss
            self.update_bar(bar=bar, loss=batch_loss)

            # scheduler step
            self.schedule(ParameterKeys.VALIDATION)

        # early stop
        self.trigger = self.early_stop_callback(cumulated_loss=cumulated_loss)

        # end epoch validation
        cumulated_loss = cumulated_loss / len(self.val_loader)
        self.print_stats(
            epoch=epoch,
            cumulated_loss=cumulated_loss,
            phase=ParameterKeys.VALIDATION,
        )

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        if not self.test_loader:
            return {}
        self.load_state_dict()

        bar = self.get_bar(self.test_loader, desc="Test")
        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            gts = data_dict[DataDict.GTS].to(self.device)

            out = self.model(imgs, self.graph.x_dict, self.graph.edge_index_dict)
            for m in metrics:
                metrics[m].update(out, gts)
        return {k: v.compute().cpu().item() for k, v in metrics.items()}

    def launch(self):
        print("Start experiment!")
        for p in self.model.vit.parameters():
            p.requires_grad = False
        for p in self.model.vit.head.parameters():
            p.requires_grad = True

        self.model = self.model.to(self.device)

        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break

        return self.test()
