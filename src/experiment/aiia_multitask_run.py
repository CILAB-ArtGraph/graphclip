from src.models import CLIP
from .clip_multitask_run import CLIPMultitaskRun
from src.models import AIxIAMultiTaskModel
from copy import deepcopy
from .utils import ParameterKeys
from timm import create_model
from torch_geometric.nn.models.basic_gnn import BasicGNN
import src.models.graph as graph_module
from torch_geometric.data import HeteroData
from src.data import DataDict
from typing import Any
import torch


class AIxIAMultiTaskRun(CLIPMultitaskRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_model(self) -> AIxIAMultiTaskModel:
        model_params = deepcopy(self.parameters).get(ParameterKeys.MODEL)
        visual_params = model_params.get(ParameterKeys.VISUAL)
        checkpoint_pth = visual_params.pop(ParameterKeys.CHECKPOINT, None)
        vit = create_model(**visual_params)
        if checkpoint_pth:
            state_dict = torch.load(checkpoint_pth, map_location=self.device)
            print("loading state dict for vit")
            vit.load_state_dict(state_dict)
            vit.reset_classifier(num_classes=0)
        gnn_model = self.__init_gnn()
        return AIxIAMultiTaskModel(
            vit=vit,
            gnn=gnn_model,
            metadata=self.graph.metadata(),
            **model_params.get(ParameterKeys.PARAMS, {}),
        ).to(self.device)

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

        metrics = deepcopy(self.metrics)
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            self.optimizer.zero_grad()
            out = self.model(images, self.graph.x_dict, self.graph.edge_index_dict)

            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
            }
            loss = sum([loss_out[task] * self.l[task] for task in self.task])
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.cpu().item()

            cumulated_loss = cumulated_loss + batch_loss
            # metrics
            for task in self.task:
                for m in metrics[task]:
                    metrics[task][m].update(out[task].cpu(), gts[task].cpu())

            self.update_bar(bar=bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAINING)

        cumulated_loss = cumulated_loss / len(self.train_loader)
        self.print_stats(
            epoch=epoch,
            cumulated_loss=cumulated_loss,
            phase=ParameterKeys.TRAINING,
            metrics=metrics,
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        cumulated_loss = 0.0
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VALIDATION} at epoch {epoch}/{self.num_epochs}",
        )
        metrics = deepcopy(self.metrics)

        self.model.eval()
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            out = self.model(images, self.graph.x_dict, self.graph.edge_index_dict)
            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
            }
            loss = sum([loss_out[task] * self.l[task] for task in self.task])
            batch_loss = loss.cpu().item()

            cumulated_loss = cumulated_loss + batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
            for task in self.task:
                for m in metrics[task]:
                    metrics[task][m].update(out[task].cpu(), gts[task].cpu())

        self.schedule(ParameterKeys.VALIDATION, metrics=cumulated_loss)
        self.trigger = self.early_stop_callback(cumulated_loss=cumulated_loss)
        cumulated_loss = cumulated_loss / len(self.val_loader)
        self.print_stats(
            epoch=epoch,
            cumulated_loss=cumulated_loss,
            phase=ParameterKeys.VALIDATION,
            metrics=metrics,
        )

    @torch.no_grad()
    def test(self) -> dict[dict[str, float], Any]:
        if not self.test_loader:
            return {}
        self.load_state_dict()

        bar = self.get_bar(self.test_loader, desc="Test")
        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            out = self.model(imgs, self.graph.x_dict, self.graph.edge_index_dict)
            for task in self.task:
                for m in metrics[task]:
                    metrics[task][m].update(out[task].cpu(), gts[task].cpu())
        return {
            task: {k: v.compute().cpu().item() for k, v in metric.items()}
            for task, metric in metrics.items()
        }

    def launch(self):
        print("Start experiment!")
        for p in self.model.vit.parameters():
            p.requires_grad = False

        self.model = self.model.to(self.device)

        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break

        return self.test()
