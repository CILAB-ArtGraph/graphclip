from typing import Any, Optional, Union
from pandas import DataFrame
from torchvision.transforms import Compose
from src.data.utils import DataDict
import nlpaug.augmenter.word as naw
from .image_multitask_dataset import ImageMultiClassDataset
from copy import deepcopy
import torch


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
        data_dict.update({DataDict.TEXT: gts})
        return data_dict


class CLIPMultiTaskDataset(ImageTextMultiTaskDataset):
    def __init__(
        self,
        dataset: str | DataFrame,
        img_dir: str,
        tasks: list[str],
        sources: dict[str, Union[str, DataFrame]],
        preprocess: Compose | dict | str | None = None,
        preprocess_classes: Optional[dict] = None,
        source_col: Optional[str] = DataDict.STYLE,
        **kwargs
    ):
        super().__init__(dataset, img_dir, tasks, preprocess, **kwargs)
        self.source_col = source_col
        self.sources = {k: self._init_dataset(v) for k, v in sources.items()}
        self.preprocess_cls = self._init_cls_preprocess(preprocess_classes)

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
        print("im here")
        images = [x[DataDict.IMAGE] for x in batched_input]
        classes = [x[DataDict.TEXT] for x in batched_input]

        batch_images = torch.stack(images)

        # dict
        # task
        #  - list[idxs]
        classes_idxs = {
            task: [self.class2idx[task][x[task]] for x in classes]
            for task in classes[0].keys()
        }
        print(classes_idxs)
        unique_classes = {k: list(set(v)) for k, v in classes_idxs.items()}
        class2batch = {
            task: {v: k for k, v in enumerate(x)} for task, x in unique_classes.items()
        }
        
        print(class2batch)

        summaries = {
            task: self.sources[task]
            .loc[unique_classes[task], DataDict.SUMMARY]
            .tolist()
            for task in self.tasks
        }

        aug_summaries = {
            task: self.augment_texts(summary) for task, summary in summaries.items()
        }

        gts = {
            task: torch.as_tensor([class2batch[task][x] for x in classes_idxs[task]])
            for task in self.tasks
        }

        return {
            DataDict.IMAGE: batch_images,
            DataDict.TEXT: aug_summaries,
            DataDict.GTS: gts,
        }
