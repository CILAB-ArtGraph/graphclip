from typing import Any
from pandas import DataFrame
from torchvision.transforms import Compose

from src.data.utils import DataDict
from .image_multitask_dataset import ImageMultiClassDataset


class ImageTextMultiTaskDataset(ImageMultiClassDataset):
    def __init__(
        self,
        dataset: str | DataFrame,
        img_dir: str,
        tasks: list[str],
        preprocess: Compose | dict | str | None = None,
        **kwargs
    ):
        super().__init__(dataset, img_dir, tasks, preprocess, **kwargs)
        self.idx2class = {
            task: {v: k for k, v in self.class2idx[task].items()}
            for task in self.class2idx.keys()
        }

    def __getitem__(self, index) -> dict[DataDict, Any]:
        data_dict = super().__getitem__(index)
        gts = data_dict.pop(DataDict.GTS)
        gts = {k: self.idx2class[k][v] for k, v in gts.items()}
        data_dict.update({DataDict.GTS: gts})
        return data_dict