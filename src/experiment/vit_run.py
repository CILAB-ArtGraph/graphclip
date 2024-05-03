from torch.optim.lr_scheduler import LambdaLR
from src.models import ViT
from .clip_run import CLIPRun
from src.data import DataDict
from .utils import ParameterKeys
import torch
from copy import deepcopy
from open_clip import create_model


class ViTRun(CLIPRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_model(self) -> ViT:
        clip_model = create_model(
            **self.parameters.get(ParameterKeys.MODEL).get(ParameterKeys.MODEL)
        )
        self.model = ViT(
            model=clip_model,
            **self.parameters.get(ParameterKeys.MODEL).get(ParameterKeys.PARAMS),
        ).to(self.device)
        self.load_state_dict()
        return self.model

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
            out = self.model(images)
            loss = self.criterion(out, gts)
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
            gts = data_dict[DataDict.GTS].to(self.device)

            out = self.model(images)
            loss = self.criterion(out, gts)
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

    def launch(self):
        print("Start experiment!")
        self.model = self.model.to(self.device)
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break

        # test
        return self.test()

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        # check for test loader
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)

        bar = self.get_bar(self.test_loader, desc="Test")

        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            gts = data_dict[DataDict.GTS]

            out = self.model(imgs)
            out = out.detach().cpu()
            for m in metrics:
                metrics[m].update(out, gts)
        return {k: v.compute().cpu().item() for k, v in metrics.items()}
