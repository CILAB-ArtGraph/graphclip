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


class CLIPRun(Run):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.init()

    def init(self):
        # init dataloaders
        self.train_loader, self.val_loader, self.test_loader = self._init_dataloaders()
        # init clip model
        self.model = self._init_model()
        # init metrics
        self.metrics = self._init_metrics()
        # init tokenizer
        self.tokenizer = self._init_tokenizer()
        # init loss
        self.loss = self._init_loss()
        # init optimizer
        self.optimizer = self._init_optimizer()
        # init scheduler
        self.scheduler = self._init_scheduler()
        # init early stop
        self.early_stop = self._init_early_stop()
        # init other parameters
        self._init_general()

    def _init_model(self) -> CLIP:
        return create_model(**self.parameters.get(ParameterKeys.MODEL))

    def _init_tokenizer(self) -> SimpleTokenizer:
        return get_tokenizer(**self.parameters.get(ParameterKeys.TOKENIZER, {}))

    def _init_loss(self) -> torch.nn.Module:
        params = deepcopy(self.parameters.get(ParameterKeys.LOSS), None)
        if not params:
            return None
        loss_name = params.get(ParameterKeys.NAME)
        loss_params = params.get(ParameterKeys.PARAMS, {})
        return losses.__dict__[loss_name](**loss_params)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None, "Please first initialize the model!!"
        params = deepcopy(self.parameters.get(ParameterKeys.OPTIMIZER), None)
        if not params:
            return None
        opt_name = params.get(ParameterKeys.NAME)
        opt_params = params.get(ParameterKeys.PARAMS, {})
        return torch.optim.__dict__[opt_name](
            parameters=self.model.parameters(), **opt_params
        )

    def _init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        assert self.optimizer is not None or (
            not self.parameters.get(ParameterKeys.OPTIMIZER),
            None,
        ), "Please first initialize the model!!"
        params = deepcopy(self.parameters.get(ParameterKeys.SCHEDULER), None)
        if not params:
            return None
        scheduler_name = params.get(ParameterKeys.NAME)
        scheduler_params = params.get(ParameterKeys.PARAMS)
        return torch.optim.lr_scheduler.__dict__[scheduler_name](**scheduler_params)

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
        self.warmup_lblock_epochs = self.parameters.get(
            ParameterKeys.WARMUP_EPOCHS, ParameterKeys.DEF_WARMUP_EPOCHS
        )
        
        
    def train_epoch(self, epoch):
        pass
    
    @torch.no_grad()
    def validate_epoch(self, epoch):
        pass
    
    def launch(self):
        pass
    

    def get_prompts(
        self, classes: list[str], source: str = None, key: str = None
    ) -> list[str]:
        if not source:
            return [f"This artwork is in {x} artistic style" for x in classes]
        key = key if key else "style"
        styles = pd.read_csv(source).set_index(key)
        return [styles.loc[x].summary for x in classes]

    def get_class_maps(self, classes, source: str = None, key: str = None):
        classes = list(sorted(list(set(deepcopy(classes)))))
        idx2class = {ix: s for ix, s in enumerate(classes)}
        class2idx = {s: ix for ix, s in idx2class.items()}

        class_prompts = self.get_prompts(classes=classes, source=source, key=key)

        return class_prompts, idx2class, class2idx

    def get_classes_for_test(self):
        dataset = self.test_loader.dataset
        return dataset.dataset[dataset.dataset.columns[-1]].unique().tolist()

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        self.model = self.model.to(self.device)
        classes = self.get_classes_for_test()
        class_source = self.parameters.get(ParameterKeys.CLASS_SOURCE, None)
        class_key = self.parameters.get(ParameterKeys.KEY, None)
        class_prompts, idx2class, class2idx = self.get_class_maps(
            classes, source=class_source, key=class_key
        )

        print(f"Having {len(classes)} classes")
        class_tokens = self.tokenizer(class_prompts).to(self.device)

        class_feats = self.model.encode_text(class_tokens)
        class_feats /= class_feats.norm(dim=-1, keepdim=True)

        bar = self.get_bar(self.test_loader, desc="Test")

        metrics = deepcopy(self.metrics)

        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(self.device)
            txts = data_dict[DataDict.TEXT]
            labels = torch.as_tensor(list(map(lambda x: class2idx[x], txts))).float()
            img_feats = self.model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            out = img_feats @ class_feats.T
            out = out.detach().cpu()
            for m in metrics:
                metrics[m].update(out, labels)
        return {k: v.compute().cpu().item() for k, v in metrics.items()}
