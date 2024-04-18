from .image_text_dataset import ImageTextDatset
from typing import Any, Union, Optional
import pandas as pd
from torchvision.transforms import Compose
from .utils import DataDict
import torch


class ImgeGraphDataset(ImageTextDatset):
    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        img_dir: str,
        graph_style_mapping: Union[str, pd.DataFrame],
        preprocess: Optional[Union[str, dict, Compose]] = None,
        mapping_kwargs: Optional[dict] = {},
        **kwargs,
    ) -> None:
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.graph_style_mapping = self._init_dataset(
            dataset=graph_style_mapping, **mapping_kwargs
        )
        assert (
            DataDict.STYLE in self.graph_style_mapping.columns
        ), f"{DataDict.STYLE} must be in mapping columns"

        self.idx2style = {
            k: v[DataDict.STYLE] for k, v in self.graph_style_mapping.iterrows()
        }
        self.style2idx = {v: k for k, v in self.idx2style.items()}

    def collate_fn(self, batched_input: list[dict[str, Any]]) -> dict[str, Any]:
        # get all images and classes in the batch
        images = [x[DataDict.IMAGE] for x in batched_input]
        classes = [x[DataDict.TEXT] for x in batched_input]

        # collate images
        batch_images = torch.stack(images)

        # collate classes
        classes_idxs = [self.style2idx[x] for x in classes]
        unique_classes = list(set(classes_idxs))
        class2batch = {v: k for k, v in enumerate(unique_classes)}

        # get gts
        gts = torch.as_tensor([class2batch[x] for x in classes_idxs])

        return {
            DataDict.IMAGE: batch_images,
            DataDict.NODES: unique_classes,
            DataDict.GTS: gts,
        }
