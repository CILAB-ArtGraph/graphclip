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
from copy import deepcopy


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
        target: Union[str, pd.DataFrame],
        img_dir: str,
        preprocess: Optional[Union[Compose, dict, str]] = None,
        preprocess_classes: Optional[dict] = None,
        target_col: Optional[str] = DataDict.STYLE,
        **kwargs,
    ):
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.target = self._init_dataset(dataset=target, **kwargs)
        self.target_col = target_col
        self.target_col = self.get_target_col_name()
        self.idx2class = {k: v for k, v in self.target[self.target_col].items()}
        self.class2idx = {v: k for k, v in self.idx2class.items()}
        self.preprocess_cls = self._init_cls_preprocess(preprocess_classes)

    def get_target_col_name(self):
        if isinstance(self.target_col, str):
            return self.target_col
        assert isinstance(self.target_col, int)
        return self.target.columns[self.target_col]

    def _init_cls_preprocess(self, cls_preprocess: Optional[dict] = None) -> list:
        if not cls_preprocess:
            return None
        augmentations = naw.__dict__
        out = []
        for entry, val in cls_preprocess.items():
            out.append(augmentations[entry](**val).augment)
        return out
    
    def augment_texts(self, texts: list[str]) -> list[str]:
        if not self.preprocess_cls:
            return texts
        aug_txt = deepcopy(texts)
        for aug in self.preprocess_cls:
            aug_txt = aug(aug_txt)
        return aug_txt

    def collate_fn(self, batched_input: list[dict[str, Any]]) -> dict[str, Any]:
        # INPUT -> (images, texts)
        # OUTPUT -> (images, texts, gts)
        images = [x[DataDict.IMAGE] for x in batched_input]
        classes = [x[DataDict.TEXT] for x in batched_input]
        
        # collect images
        batch_images = torch.stack(images)
        
        # collate texts
        classes_idxs = [self.class2idx[x] for x in classes]
        unique_classes = list(set(classes_idxs))
        class2batch = {v: k for k, v in enumerate(unique_classes)}
        
        # get summaries
        summaries = self.target.loc[unique_classes, DataDict.SUMMARY].tolist()
        
        # augment summaries
        aug_summaries = self.augment_texts(summaries)
        
        # get gts
        gts = torch.as_tensor([class2batch[x] for x in classes_idxs])#.float()
        
        return {
            DataDict.IMAGE: batch_images,
            DataDict.TEXT: aug_summaries,
            DataDict.GTS: gts,
        }


if __name__ == "__main__":
    dataset = ImageTextDatset(
        dataset="data/processed/normal/artgraph_clip/test.csv",
        img_dir="data/raw/images-resized",
    )

    print(dataset[0])
