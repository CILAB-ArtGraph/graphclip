from .image_text_dataset import ImageTextDatset
from typing import Any, Union, Optional
import pandas as pd
from torchvision.transforms import Compose
from .utils import DataDict
import torch


class ImageGraphDataset(ImageTextDatset):
    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        img_dir: str,
        graph_class_mapping: Union[str, pd.DataFrame],
        mapping_target_col: Union[str, int],
        preprocess: Optional[Union[str, dict, Compose]] = None,
        mapping_kwargs: Optional[dict] = {},
        **kwargs,
    ) -> None:
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.graph_class_mapping = self._init_dataset(
            dataset=graph_class_mapping, **mapping_kwargs
        )
        self.mapping_target_col = mapping_target_col

        self.idx2class = {
            k: v[self.mapping_target_col]
            for k, v in self.graph_class_mapping.iterrows()
        }
        self.class2idx = {v: k for k, v in self.idx2class.items()}

    def collate_fn(self, batched_input: list[dict[str, Any]]) -> dict[str, Any]:
        # get all images and classes in the batch
        images = [x[DataDict.IMAGE] for x in batched_input]
        classes = [x[DataDict.TEXT] for x in batched_input]

        # collate images
        batch_images = torch.stack(images)

        # collate classes
        classes_idxs = [self.class2idx[x] for x in classes]
        unique_classes = list(set(classes_idxs))
        class2batch = {v: k for k, v in enumerate(unique_classes)}

        # get gts
        gts = torch.as_tensor([class2batch[x] for x in classes_idxs])

        return {
            DataDict.IMAGE: batch_images,
            DataDict.NODES: unique_classes,
            DataDict.GTS: gts,
        }
