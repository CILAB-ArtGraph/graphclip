from typing import Any
from .run import Run
from src.models import CLIP
from open_clip import create_model, get_tokenizer, SimpleTokenizer
from .utils import ParameterKeys
from copy import deepcopy
import torch
from src.data import DataDict
import pandas as pd
import src.loss as losses
from src.utils import early_stop
from typing import Optional
import torch.optim.lr_scheduler as schedulers
from src.scheduler import scheduler_registry
import os


class CLIPRun(Run):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.init()

    def init(self):
        # init other parameters
        print("Loading general parameters...")
        self._init_general()
        print("Done!")

        # init dataloaders
        print("Loading dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = self._init_dataloaders()
        print("Done!")

        # init clip model
        print("Loading model...")
        self.model = self._init_model()
        print("Done!")

        # init metrics
        print("Loading metrics...")
        self.metrics = self._init_metrics()
        print("Done!")

        # init tokenizer
        print("Loading tokenizer...")
        self.tokenizer = self._init_tokenizer()
        print("Done!")

        # init loss
        print("Loading criterion...")
        self.criterion = self._init_criterion()
        print("Done!")

        # init optimizer
        print("Loading optimizer...")
        self.optimizer = self._init_optimizer()
        print("Done!")

        # init scheduler
        print("Loading scheduler...")
        self.scheduler = self._init_scheduler()
        print("Done!")

        # init early stop
        print("Loading early stop callback...")
        self.early_stop = self._init_early_stop()
        print("Done!")

    def _init_model(self) -> CLIP:
        model = create_model(**self.parameters.get(ParameterKeys.MODEL)).to(self.device)
        self.load_state_dict()
        return model

    def _init_tokenizer(self) -> SimpleTokenizer:
        tokenizer_params = self.parameters.get(ParameterKeys.TOKENIZER, {})
        if not tokenizer_params:
            return None
        return get_tokenizer(**tokenizer_params)

    def _init_criterion(self) -> torch.nn.Module:
        params = deepcopy(self.parameters.get(ParameterKeys.CRITERION), None)
        if not params:
            return None
        criterion_name = params.get(ParameterKeys.NAME)
        criterion_params = params.get(ParameterKeys.PARAMS, {})
        return losses.__dict__[criterion_name](**criterion_params)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None, "Please first initialize the model!!"
        params = deepcopy(self.parameters.get(ParameterKeys.OPTIMIZER), None)
        if not params:
            return None
        opt_name = params.get(ParameterKeys.NAME)
        opt_params = params.get(ParameterKeys.PARAMS, {})
        return torch.optim.__dict__[opt_name](
            params=self.model.parameters(), **opt_params
        )

    def _init_scheduler(self) -> schedulers.LRScheduler:
        assert self.optimizer is not None or (
            not self.parameters.get(ParameterKeys.OPTIMIZER),
            None,
        ), "Please first initialize the optimizer!!"
        params = deepcopy(self.parameters.get(ParameterKeys.SCHEDULER), None)
        if not params:
            return None
        scheduler_name = params.get(ParameterKeys.NAME)
        scheduler_params = params.get(ParameterKeys.PARAMS, {})
        if scheduler_name == ParameterKeys.HUGGINGFACE_COS_SCHEDULER:
            scheduler_params = self._init_cosine_scheduler_params(
                params=scheduler_params
            )
            self.cosine_sched = True
        else:
            self.cosine_sched = False
        return scheduler_registry[scheduler_name](
            optimizer=self.optimizer, **scheduler_params
        )

    def _init_cosine_scheduler_params(self, params) -> schedulers.LambdaLR:
        # get total steps
        total_steps = self.num_epochs * len(self.train_loader)
        params.update({"num_training_steps": total_steps})
        return params

    def _init_early_stop(self):
        params = deepcopy(self.parameters.get(ParameterKeys.EARLY_STOP), None)
        if not params:
            return None
        early_stop_name = params.get(ParameterKeys.NAME)
        early_stop_params = params.get(ParameterKeys.PARAMS)
        model_state_dict_path = (
            f"{self.parameters.get(ParameterKeys.OUT_DIR)}/{ParameterKeys.MODEL}.pt"
        )
        early_stop_params.update({ParameterKeys.PATH: model_state_dict_path})
        return early_stop.__dict__[early_stop_name](**early_stop_params)

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
        self.task = self.parameters.get(ParameterKeys.TASK, ParameterKeys.DEF_TASK)
        self.trigger = False

    def schedule(self, phase, **kwargs):
        if not self.scheduler:
            return
        if phase == ParameterKeys.VALIDATION and self.cosine_sched:
            return
        if phase == ParameterKeys.TRAINING and not self.cosine_sched:
            return
        self.scheduler.step(**kwargs)

    def print_stats(
        self,
        epoch: int,
        cumulated_loss: float,
        phase: str,
        metrics: Optional[dict] = None,
    ) -> None:
        print(f"Epoch {epoch}/{self.num_epochs}: {phase} loss: {cumulated_loss:.4f}")
        if metrics:
            for k, v in metrics.items():
                metric_value = v.compute().cpu().item()
                print(
                    f"Epoch {epoch}/{self.num_epochs}: {phase} {k}: {metric_value:.4f}"
                )

    def early_stop_callback(self, cumulated_loss: float) -> bool:
        if not self.early_stop:
            return False
        self.early_stop(cumulated_loss, self.model)
        return self.early_stop.early_stop

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
            text_tokens = self.tokenizer(texts=texts).to(self.device)
            gts = data_dict[DataDict.GTS].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(image=images, text=text_tokens)
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
            texts = data_dict[DataDict.TEXT]
            text_tokens = self.tokenizer(texts=texts).to(self.device)
            gts = data_dict[DataDict.GTS].to(self.device)

            out = self.model(image=images, text=text_tokens)
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

    def set_model_parameters_for_warmup(self, current_epoch: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = current_epoch > self.warmup_lblock_epochs
        # last visual block
        for p in self.model.visual.transformer.resblocks[-1].parameters():
            p.requires_grad = True
        # last text block
        for p in self.model.transformer.resblocks[-1].parameters():
            p.requires_grad = True

    def launch(self):
        print("Start experiment!")
        self.model = self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.set_model_parameters_for_warmup(current_epoch=epoch)
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break

        # test
        return self.test()

    def get_prompts(
        self, classes: list[str], source: str = None, key: str = None
    ) -> list[str]:
        if not source:
            return [f"This artwork is in {x} artistic {self.task}" for x in classes]
        key = key if key else self.task
        classes_kb = pd.read_csv(source).set_index(key)
        return [classes_kb.loc[x].summary for x in classes]

    def get_class_maps(self, classes, source: str = None, key: str = None):
        classes = list(sorted(list(set(deepcopy(classes)))))
        idx2class = {ix: s for ix, s in enumerate(classes)}
        class2idx = {s: ix for ix, s in idx2class.items()}

        class_prompts = self.get_prompts(classes=classes, source=source, key=key)

        return class_prompts, idx2class, class2idx

    def get_classes_for_test(self):
        dataset = self.test_loader.dataset
        return dataset.dataset[dataset.dataset.columns[-1]].unique().tolist()

    def load_state_dict(self):
        out_dir = self.parameters.get(ParameterKeys.OUT_DIR)
        if os.path.exists(f"{out_dir}/ParameterKeys.MODEL.pt"):
            print("loading state dict")
            state_dict = torch.load(
                f"{out_dir}/ParameterKeys.MODEL.pt",
                map_location=self.device,
            )
            self.model.load_state_dict(state_dict=state_dict)
        else:
            print("no state dict found")

    def get_class_info(self):
        classes = self.get_classes_for_test()
        class_source = self.parameters.get(ParameterKeys.CLASS_SOURCE, None)
        class_key = self.parameters.get(ParameterKeys.KEY, None)
        class_prompts, idx2class, class2idx = self.get_class_maps(
            classes, source=class_source, key=class_key
        )
        return class_prompts, idx2class, class2idx

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        # check for test loader
        if not self.test_loader:
            return {}
        self.load_state_dict()
        self.model = self.model.to(self.device)
        classes = self.get_classes_for_test()
        class_source = self.parameters.get(ParameterKeys.CLASS_SOURCE, None)
        class_key = self.parameters.get(ParameterKeys.KEY, None)
        class_prompts, idx2class, class2idx = self.get_class_maps(
            classes, source=class_source, key=class_key
        )

        print(f"Having {len(classes)} classes")
        class_tokens = self.tokenizer(class_prompts).to(self.device)

        class_feats = self.model.encode_text(class_tokens, normalize=True)

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
