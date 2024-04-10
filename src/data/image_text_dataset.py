from torch.utils.data import Dataset
from typing import Union, Any, Optional
import pandas as pd
from torchvision.transforms import Compose
from .utils import DataDict
from PIL import Image
import torch
from torchvision.transforms import Compose
import torchvision.transforms
from open_clip import create_model_and_transforms
import nlpaug.augmenter.word as naw


class ImageTextDatset(Dataset):
    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        img_dir: str,
        preprocess: Optional[Union[Compose, dict, str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.dataset = self._init_dataset(dataset, **kwargs)
        self.img_dir = img_dir
        self.preprocess = self._init_preprocess(preprocess=preprocess)

    def _init_preprocess(self, preprocess):
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

    def _init_dataset(self, dataset, **kwargs) -> pd.DataFrame:
        return pd.read_csv(dataset, **kwargs) if isinstance(dataset, str) else dataset

    def _load_image(self, image_fname) -> Union[torch.Tensor, Image.Image]:
        return Image.open(f"{self.img_dir}/{image_fname}").convert("RGB")

    def __getitem__(self, index) -> dict[DataDict, Any]:

        image_name, class_name = self.dataset.iloc[index].tolist()
        image = self._load_image(image_name)
        image = self.preprocess(image) if self.preprocess else image
        return {DataDict.IMAGE.value: image, DataDict.TEXT.value: class_name}

    def __len__(self):
        return len(self.dataset)


class CLIPDataset(ImageTextDatset):
    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        styles: Union[str, pd.DataFrame],
        img_dir: str,
        preprocess: Optional[Union[Compose, dict, str]] = None,
        preprocess_classes: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.style = self._init_dataset(dataset=styles, **kwargs)

    def _init_cls_preprocess(self, cls_preprocess: Optional[dict] = None) -> list:
        if not cls_preprocess:
            return None
        augmentations = naw.__dict__
        out = []
        for entry, val in cls_preprocess.items():
            out.append(augmentations[entry](**val).augment)
        return out
    
    def augment_texts(self, texts: list[str]) -> list[str]:
        pass

    def collate_fn(self, batched_input: list[dict[str, Any]]) -> dict[str, Any]:
        pass


if __name__ == "__main__":
    dataset = ImageTextDatset(
        dataset="data/processed/normal/artgraph_clip/test.csv",
        img_dir="data/raw/images-resized",
    )

    print(dataset[0])
