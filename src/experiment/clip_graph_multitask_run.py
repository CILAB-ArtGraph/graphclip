from .clip_multitask_run import CLIPMultitaskRun
from copy import deepcopy
from src.models import CLIPGraphMultiTask
from .utils import ParameterKeys
from open_clip import create_model
from torch_geometric.nn.models.basic_gnn import BasicGNN
import src.models.graph as graph_module
from torch_geometric.data import HeteroData
import torch
from src.data import DataDict
import pandas as pd


class CLIPGraphMultitaskRun(CLIPMultitaskRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_general(self):
        super()._init_general()
        self.graph = self._init_graph()
        self.test_graph = self._init_test_graph()
        self.class2graphidx, self.graphidx2class = self._init_test_mapping()

    def _init_test_mapping(self):
        test_mapping_params = deepcopy(
            self.parameters.get(ParameterKeys.TEST_MAPPING, None)
        )
        if test_mapping_params is None:
            return {}, {}
        graph_class_mapping_params = test_mapping_params.get("graph_class_mapping")
        mapping_kwargs = test_mapping_params.get("mapping_kwargs")
        mapping_target_col = test_mapping_params.get("mapping_target_col")
        graph_class_mapping = {
            task: pd.read_csv(
                graph_class_mapping_params[task], **mapping_kwargs.get(task, {})
            )
            for task in graph_class_mapping_params.keys()
        }
        graph_idx2class = {
            task: {k: v[mapping_target_col.get(task)] for k, v in mapping.iterrows()}
            for task, mapping in graph_class_mapping.items()
        }
        class2graphidx = {
            task: {v: k for k, v in task_mapping.items()}
            for task, task_mapping in graph_idx2class.items()
        }
        return class2graphidx, graph_idx2class

    def _init_test_graph(self):
        graph_params = deepcopy(
            self.parameters.get(ParameterKeys.TEST_CLASS_SOURCE, {})
        )
        return torch.load(**graph_params) if graph_params else None

    def _init_model(self) -> CLIPGraphMultiTask:
        model_params = deepcopy(self.parameters).get(ParameterKeys.MODEL)
        clip_model = create_model(**model_params.get(ParameterKeys.VISUAL))
        gnn_model = self.__init_gnn()
        return CLIPGraphMultiTask(
            clip_model=clip_model,
            gnn_model=gnn_model,
            metadata=(
                self.test_graph.metadata()
                if self.test_graph is not None
                else self.graph.metadata()
            ),
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

    def set_model_parameters_for_warmup(self, current_epoch: int) -> None:
        # start setting ViT parameters freezed
        for p in self.model.visual.parameters():
            p.requires_grad = False

        # let always gnn parameters to be twiched
        for p in self.model.gnn.parameters():
            p.requires_grad = True

        if current_epoch < self.warmup_lblock_epochs:
            print("Warm-up phase. ViT completely frozen")
            return
        start_block = (len(self.model.visual.transformer.resblocks) - 1) - (
            (current_epoch - self.warmup_lblock_epochs) // 2
        )
        start_block = start_block if start_block > 0 else 0
        print(
            f"Unlocking {len(self.model.visual.transformer.resblocks[start_block: ])}/{len(self.model.visual.transformer.resblocks)} layers for ViT"
        )
        for p in self.model.visual.transformer.resblocks[start_block:].parameters():
            p.requires_grad = True

    def train_epoch(self, epoch):
        cumulated_loss = 0.0
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAINING} at epoch {epoch}/{self.num_epochs}",
        )

        self.model.train()
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            nodes = data_dict[DataDict.NODES]
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            self.optimizer.zero_grad()
            out = self.model(
                image=images,
                x_dict=self.graph.x_dict,
                edge_index_dict=self.graph.edge_index_dict,
                target_nodes=nodes,
                return_dict=True,
            )
            loss_out = {
                task: self.criterion[task](out, gts[task]) for task in self.task
            }
            loss = sum([loss_out[task]["loss"] * self.l[task] for task in self.task])
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.cpu().item()

            # update stats
            cumulated_loss = cumulated_loss + batch_loss
            self.update_bar(bar=bar, loss=batch_loss)

            # scheduler step
            self.schedule(phase=ParameterKeys.TRAINING)

        # end epoch training
        cumulated_loss = cumulated_loss / len(self.train_loader)
        self.print_stats(
            epoch=epoch,
            cumulated_loss=cumulated_loss,
            phase=ParameterKeys.TRAINING,
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
            nodes = data_dict[DataDict.NODES]
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            out = self.model(
                image=images,
                x_dict=self.graph.x_dict,
                edge_index_dict=self.graph.edge_index_dict,
                target_nodes=nodes,
                return_dict=True,
            )
            loss_out = {
                task: self.criterion[task](out, gts[task]) for task in self.task
            }
            loss = sum([loss_out[task]["loss"] * self.l[task] for task in self.task])
            batch_loss = loss.cpu().item()

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

    def get_class_maps(self):
        return (
            (self.class2graphidx, self.graphidx2class)
            if self.class2graphidx is not None
            else (
                self.train_loader.dataset.class2graphidx,
                self.train_loader.dataset.graphidx2class,
            )
        )

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)
        class2idx, idx2class = self.get_class_maps()

        for task in self.task:
            print(f"Having {len(list(class2idx[task].keys()))} classes for task {task}")

        graph = self.test_graph if self.test_graph else self.graph

        class_feats = self.model.encode_graph(
            graph.x_dict, graph.edge_index_dict, normalize=True
        )

        bar = self.get_bar(self.test_loader, desc="Test")

        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            txts = data_dict[DataDict.TEXT]
            labels = {
                task: torch.as_tensor(
                    list(map(lambda x: class2idx[task][x], cls_txt))
                ).float()
                for task, cls_txt in txts.items()
            }
            img_feats = self.model.encode_image(imgs, normalize=True)

            for task in self.task:
                out = img_feats @ class_feats[task].T
                out = out.detach().cpu()
                for m in metrics[task]:
                    metrics[task][m].update(out, labels[task])
        return {
            task: {k: v.compute().cpu().item() for k, v in metric.items()}
            for task, metric in metrics.items()
        }
