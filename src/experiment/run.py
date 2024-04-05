import src.models
import src.data
from .utils import ParameterKeys
from copy import deepcopy
import torch
from typing import Any
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm


class Run:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_bar(self, loader, desc):
        if not self.parameters.get(ParameterKeys.BAR, False):
            return loader
        return tqdm(
            enumerate(loader),
            total=len(loader),
            desc=desc,
        )

    def _init_model(self) -> Any:
        models = src.models.__dict__
        model_params = deepcopy(self.parameters[ParameterKeys.MODEL])
        model_name = model_params.pop(ParameterKeys.NAME)
        state_dict = model_params.pop(ParameterKeys.STATE_DICT, None)
        model = models[model_name](**model_params[ParameterKeys.PARAMS])
        if state_dict:
            model.load_state_dict(torch.load(state_dict))
        return model

    def _init_dataset(self, params):
        dataset_types = src.data.__dict__
        if not params:
            return None
        dataset_name = params.pop(ParameterKeys.NAME)
        return dataset_types[dataset_name](**params.get(ParameterKeys.PARAMS, {}))

    def _init_loader(self, dataset, loader_params):
        loader_params_copy = deepcopy(loader_params)
        if dataset is None:
            return None
        if hasattr(dataset, "collate_fn"):
            loader_params_copy["collate_fn"] = dataset.collate_fn
        return DataLoader(dataset, **loader_params)

    def _init_dataloaders(self):
        dataset_params = deepcopy(self.parameters[ParameterKeys.DATASET])
        loader_params = self.parameters.get(ParameterKeys.DATALOADER, {})

        train_params = dataset_params.get(ParameterKeys.TRAIN, None)
        train_data = self._init_dataset(train_params)

        val_params = deepcopy(dataset_params.get(ParameterKeys.VAL, None))
        val_data = self._init_dataset(val_params)

        test_params = deepcopy(dataset_params.get(ParameterKeys.TEST, None))
        test_data = self._init_dataset(test_params)

        return (
            self._init_loader(train_data, loader_params),
            self._init_loader(val_data, loader_params),
            self._init_loader(test_data, loader_params),
        )

    def _init_metrics(self):
        metrics_types = torchmetrics.__dict__
        metrics_params = deepcopy(self.parameters.get(ParameterKeys.METRICS))
        return {k: metrics_types[k](**v) for k, v in metrics_params.items()}

    def train_epoch(self, epoch):
        raise NotImplementedError()

    def validate_epoch(self, epoch):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
