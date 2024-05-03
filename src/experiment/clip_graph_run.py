from .clip_run import CLIPRun
from src.models import CLIPGraph
from open_clip import create_model_and_transforms
from open_clip import create_model
from copy import deepcopy
from .utils import ParameterKeys
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import HeteroData
import torch
from src.data.utils import DataDict
import src.models.graph as graph_module


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
            gts = data_dict[DataDict.GTS].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(
                image=images,
                x_dict=self.graph.x_dict,
                edge_index_dict=self.graph.edge_index_dict,
                target_nodes=nodes,
                return_dict=True,
            )
            loss_out = self.criterion(out, gts)
            loss = loss_out["loss"]
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
            gts = data_dict[DataDict.GTS].to(self.device)

            out = self.model(
                image=images,
                x_dict=self.graph.x_dict,
                edge_index_dict=self.graph.edge_index_dict,
                target_nodes=nodes,
                return_dict=True,
            )
            loss_out = self.criterion(out, gts)
            loss = loss_out["loss"]
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
        return self.train_loader.dataset.class2idx, self.train_loader.dataset.idx2class

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)
        classes = self.get_classes_for_test()
        class2idx, idx2class = self.get_class_maps()

        print(f"Having {len(classes)} classes")

        class_feats = self.model.encode_graph(
            self.graph.x_dict, self.graph.edge_index_dict, normalize=True
        )

        bar = self.get_bar(self.test_loader, desc="Test")

        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            txts = data_dict[DataDict.TEXT]
            labels = torch.as_tensor(list(map(lambda x: class2idx[x], txts))).float()
            img_feats = self.model.encode_image(imgs, normalize=True)

            out = img_feats @ class_feats.T
            out = out.detach().cpu()
            for m in metrics:
                metrics[m].update(out, labels)
        return {k: v.compute().cpu().item() for k, v in metrics.items()}
