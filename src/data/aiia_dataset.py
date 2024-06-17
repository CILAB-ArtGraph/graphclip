from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import Compose
import torchvision
from open_clip import create_model_and_transforms
import torch
from PIL import Image
from .utils import DataDict


class AIxIADataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        dataset: str | pd.DataFrame,
        mapping: str | pd.DataFrame,
        preprocess: Compose | dict = None,
        mapping_kwargs={},
        data_kwargs={},
    ) -> None:
        self.img_dir = img_dir
        self.dataset = self.init_dataset(dataset, **data_kwargs)
        self.preprocess = self.init_preprocess(preprocess)
        self.mapping = self.init_dataset(mapping, **mapping_kwargs)
        self.mapping_dict = {val[1]: val[0] for _, val in self.mapping.iterrows()}

    def _load_image(self, image_fname) -> Image.Image:
        return Image.open(f"{self.img_dir}/{image_fname}").convert("RGB")

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        image_name, class_name = self.dataset.iloc[index].tolist()
        image = self._load_image(image_name)
        image = self.preprocess(image) if self.preprocess else image
        label = self.mapping_dict[class_name]
        return {DataDict.IMAGE: image, DataDict.GTS: torch.as_tensor([label])}

    def __len__(self):
        return len(self.dataset)

    def init_dataset(self, dataset: str | pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.read_csv(dataset, **kwargs) if isinstance(dataset, str) else dataset

    def init_preprocess(self, preprocess):
        if not preprocess:
            return None
        if isinstance(preprocess, Compose):
            return preprocess
        if preprocess.get("name").lower() == "clip":
            return create_model_and_transforms(**preprocess.get("params"))[2]

        assert preprocess.get("name").lower() == "compose"
        prep_types = torchvision.transforms.__dict__
        return Compose(
            [
                prep_types[k](**(v if isinstance(v, dict) else {}))
                for k, v in preprocess.get("params").items()
            ]
        )
