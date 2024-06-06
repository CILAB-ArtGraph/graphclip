from torch.nn.modules import Module
from .clip_run import CLIPRun
from .utils import ParameterKeys
import torchmetrics
from copy import deepcopy
import src.loss as losses
from src.data import DataDict
import torch
import os


class CLIPMultitaskRun(CLIPRun):
    def _init_general(self):
        self.device = self.parameters.get(
            ParameterKeys.DEVICE, ParameterKeys.CPU_DEVICE
        )
        self.out_dir = self.parameters.get(
            ParameterKeys.OUT_DIR, ParameterKeys.DEF_OUT_DIR
        )
        self.clean_out_dir = self.parameters.get(ParameterKeys.CLEAN_OUT_DIR, False)
        os.makedirs(self.out_dir, exist_ok=True)
        if self.clean_out_dir:
            for f in os.listdir(self.out_dir):
                os.remove(f"{self.out_dir}/{f}")
        self.warmup_lblock_epochs = self.parameters.get(
            ParameterKeys.WARMUP_EPOCHS, ParameterKeys.DEF_WARMUP_EPOCHS
        )
        self.num_epochs = self.parameters.get(ParameterKeys.NUM_EPOCHS, 0)
        self.bar = self.parameters.get(ParameterKeys.BAR, False)
        self.task = self.parameters.get(ParameterKeys.TASK)  # now it is a list
        self.l = self.parameters.get(
            ParameterKeys.LAMBDA, {}
        )  # dict task-> lambda_task
        self.trigger = False

    def _init_metrics(self):
        metrics_types = torchmetrics.__dict__
        metrics_params = deepcopy(self.parameters.get(ParameterKeys.METRICS), {})
        return {
            task: {
                k: metrics_types[v.get(ParameterKeys.NAME)](
                    **v.get(ParameterKeys.PARAMS)
                )
                for k, v in metrics_params[task].items()
            }
            for task in self.task
        }

    def _init_criterion(self) -> dict[str, Module]:
        params = deepcopy(self.parameters.get(ParameterKeys.CRITERION), None)
        if not params:
            return None
        return {
            task: losses.__dict__[params.get(task).get(ParameterKeys.NAME)](
                **params.get(task).get(ParameterKeys.PARAMS, {})
            )
            for task in self.task
        }

    def print_stats(
        self, epoch: int, cumulated_loss: float, phase: str, metrics: dict | None = None
    ) -> None:
        print(f"Epoch {epoch}/{self.num_epochs}: {phase} loss: {cumulated_loss:.4f}")
        if metrics:
            for task in self.task:
                for k, v in metrics[task]:
                    metric_value = v.compute()
                    print(
                        f"Epoch {epoch}/{self.num_epochs}: Task: {task}, {phase} {k}: {metric_value.item():.4f}"
                    )

    def train_epoch(self, epoch):
        cumulated_loss = 0.0
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAINING} at epoch {epoch}/{self.num_epochs}",
        )

        self.model.train()
        for ix, data_dict in bar:
            images = data_dict[DataDict.IMAGE].to(self.device)
            texts = data_dict[DataDict.TEXT]
            # tokens extracted for each task
            text_tokens = {
                task: self.tokenizer(texts=texts[task]).to(self.device)
                for task in self.task
            }
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            self.optimizer.zero_grad()
            out = {
                task: self.model(image=images, text=text_tokens[task])
                for task in self.task
            }

            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
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
            texts = data_dict[DataDict.TEXT]
            # tokens extracted for each task
            text_tokens = {
                task: self.tokenizer(texts=texts[task]).to(self.device)
                for task in self.task
            }
            gts = {k: v.to(self.device) for k, v in data_dict[DataDict.GTS].items()}

            out = {
                task: self.model(image=images, text=text_tokens[task])
                for task in self.task
            }

            loss_out = {
                task: self.criterion[task](out[task], gts[task]) for task in self.task
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
        source_key = self.parameters.get(ParameterKeys.KEY)
        classes = {
            task: list(self.train_loader.dataset.sources[task].index.tolist())
            for task in self.task
        }
        idx2class = {
            task: {ix: s for ix, s in enumerate(classes[task])}
            for task in classes.keys()
        }
        class2idx = {
            task: {v: k for k, v in task_maps.items()}
            for task, task_maps in idx2class.items()
        }
        class_prompts = {
            task: self.train_loader.dataset.sources[task].loc[cls_list, source_key]
            for task, cls_list in classes.items()
        }

        return class_prompts, idx2class, class2idx

    @torch.no_grad()
    def test(self) -> dict[dict[str, float]]:
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)
        class_prompts, idx2class, class2idx = self.get_class_maps()

        for task in self.task:
            print(f"Having {len(class_prompts[task])} classes for task {task}")

        class_tokens = {
            task: self.tokenizer(prompts).to(self.device)
            for task, prompts in class_prompts.items()
        }

        class_feats = {
            task: self.model.encode_text(tokens, normalize=True)
            for task, tokens in class_tokens.items()
        }

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
