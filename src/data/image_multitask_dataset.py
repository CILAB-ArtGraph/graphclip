from typing import Any
from pandas import DataFrame
from torchvision.transforms import Compose
from src.data.utils import DataDict
from .image_text_dataset import ImageTextDatset


class ImageMultiTaskDataset(ImageTextDatset):
    def __init__(
        self,
        dataset: str | DataFrame,
        img_dir: str,
        tasks: list[str],
        preprocess: Compose | dict | str | None = None,
        **kwargs
    ):
        super().__init__(dataset, img_dir, preprocess, **kwargs)
        self.tasks = tasks # tasks to be fetched
        # columns in the df corresponding to the tasks
        self.target_cols = [
            self.dataset.columns.tolist().index(task) for task in self.tasks
        ]
        # for each task we have the mapping
        self.class2idx = {
            task: {
                name: idx
                for (idx, name) in enumerate(self.dataset[task].unique().tolist())
            }
            for task in self.tasks
        }
        # assertion
        assert all(
            [x in self.dataset.columns for x in tasks]
        ), "Target columns must be in dataset columns"

    def __getitem__(self, index) -> dict[DataDict, Any]:
        image_name, *tasks = self.dataset.iloc[index].tolist()
        image = self._load_image(image_name) # loading image 
        image = self.preprocess(image) if self.preprocess else image # preprocessing

        # getting gts for each task
        gts = {
            task: self.class2idx[task][tasks[idx]]  # get index of the classes
            for (task, idx) in zip(self.tasks, map(lambda x: x - 1, self.target_cols))
        }

        return {DataDict.IMAGE: image, DataDict.GTS: gts}
