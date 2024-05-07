from src.models import ViTMultiTask
from src.data import DataDict
from .clip_multitask_run import CLIPMultitaskRun
from open_clip import create_model
from .utils import ParameterKeys
import torch
from typing import Any
from copy import deepcopy


class ViTMultiTaskRun(CLIPMultitaskRun):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def _init_model(self) -> ViTMultiTask:
        clip_model = create_model(
            **self.parameters.get(ParameterKeys.MODEL).get(ParameterKeys.MODEL)
        )
        self.model = ViTMultiTask(
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
            gts = {
                task: gt.to(self.device) for task, gt in data_dict[DataDict.GTS].items()
            }

            self.optimizer.zero_grad()
            out = self.model(images)
            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
            }
            loss = sum([loss_out[task] * self.l[task] for task in self.task])
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
            gts = {
                task: gt.to(self.device) for task, gt in data_dict[DataDict.GTS].items()
            }

            out = self.model(images)
            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
            }
            loss = sum([loss_out[task] * self.l[task] for task in self.task])
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

    @torch.no_grad()
    def test(self) -> dict[dict[str, float], Any]:
        # check for test loader
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)

        bar = self.get_bar(self.test_loader, desc="Test")

        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            gts = {
                task: gt.to(self.device) for task, gt in data_dict[DataDict.GTS].items()
            }

            out = self.model(imgs)
            for task in self.task:
                for m in metrics[task]:
                    metrics[task][m].update(out[task], gts[task])
        return {
            task: {k: v.compute().cpu().item() for k, v in metric.items()}
            for task, metric in metrics.items()
        }

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