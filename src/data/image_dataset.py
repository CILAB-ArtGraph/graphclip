from torch.utils.data import Dataset
from typing import Any, Union, Optional
import pandas as pd
from torchvision.transforms import Compose

from src.data.utils import DataDict
from .image_text_dataset import ImageTextDatset


class ImageDataset(ImageTextDatset):
    def __init__(
        self,
        dataset: str | pd.DataFrame,
        img_dir: str,
        preprocess: Compose | dict | str | None = None,
        **kwargs
    ):
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.class2idx = {
            name: idx
            for (idx, name) in enumerate(
                self.dataset[self.dataset.columns[1]].unique().tolist()
            )
        }

    def __getitem__(self, index) -> dict[DataDict, Any]:
        data_dict = super().__getitem__(index)
        class_name = data_dict.pop(DataDict.TEXT)
        data_dict[DataDict.GTS] = self.class2idx[class_name]
        return data_dict
