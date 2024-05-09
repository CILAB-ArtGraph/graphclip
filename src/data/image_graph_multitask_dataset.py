from pandas import DataFrame
from torchvision.transforms import Compose
from src.data.utils import DataDict
from .image_text_multitask_dataset import ImageTextMultiTaskDataset
from typing import Any, Union, Optional
import torch


class ImageGraphMultiTaskDataset(ImageTextMultiTaskDataset):
    def __init__(
        self,
        dataset: str | DataFrame,
        img_dir: str,
        tasks: list[str],
        graph_class_mapping: dict[str, Union[str, DataFrame]],
        mapping_target_col: Union[str, int],
        mapping_kwargs: Optional[dict[str, Union[str, DataFrame]]] = None,
        preprocess: Compose | dict | str | None = None,
        **kwargs
    ):
        super().__init__(dataset, img_dir, tasks, preprocess, **kwargs)
        self.tasks = tasks
        mapping_kwargs = (
            mapping_kwargs if mapping_kwargs else {task: {} for task in self.tasks}
        )
        self.graph_class_mapping = {
            task: self._init_dataset(
                dataset=graph_class_mapping[task], **mapping_kwargs[task]
            )
            for task in self.tasks
        }
        self.mapping_target_col = mapping_target_col

        self.graphidx2class = {
            task: {k: v[self.mapping_target_col] for k, v in mapping.iterrows()}
            for task, mapping in self.graph_class_mapping.items()
        }

        self.class2graphidx = {
            task: {v: k for k, v in task_mapping.items()}
            for task, task_mapping in self.graphidx2class.items()
        }

    def collate_fn(self, batched_input: list[Any]) -> dict[str, Any]:
        images = [x[DataDict.IMAGE] for x in batched_input]
        classes = [x[DataDict.TEXT] for x in batched_input]
        batch_images = torch.stack(images)

        # collate classes
        classes_idxs = {
            task: [self.class2graphidx[task][x[task]] for x in classes]
            for task in self.tasks
        }
        unique_classes = {
            task: list(set(cls_idxs)) for task, cls_idxs in classes_idxs.items()
        }
        class2batch = {
            task: {v: k for k, v in enumerate(unique_cls)}
            for task, unique_cls in unique_classes.items()
        }

        # get gts
        gts = {
            task: torch.as_tensor([class2batch[task][x] for x in classes_idxs[task]])
            for task in self.tasks
        }

        return {
            DataDict.IMAGE: batch_images,
            DataDict.NODES: unique_classes,
            DataDict.GTS: gts,
        }
