from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pandas as pd
from open_clip import create_model_and_transforms
from PIL import Image
from .utils import DataDict
import torchvision


class AIxIAMultitaskDataset(Dataset):
    def __init__(
        self,
        dataset: str | pd.DataFrame,
        img_dir: str,
        tasks: list[str],
        mappings: dict[str, str | pd.DataFrame],
        mapping_kwargs: dict[str, dict] = None,
        preprocess: Compose | dict | str = None,
        dataset_kwargs: dict = {},
    ) -> None:
        self.img_dir = img_dir
        self.tasks = tasks
        self.dataset = self.init_dataset(dataset, **dataset_kwargs)
        self.preprocess = self.init_preprocess(preprocess=preprocess)
        mapping_kwargs = (
            mapping_kwargs if mapping_kwargs else {task: {} for task in self.tasks}
        )
        self.mappings = {
            task: self.init_dataset(mappings[task], **mapping_kwargs[task])
            for task in self.tasks
        }
        self.mappings_dict = {
            task: {val.iloc[1]: val.iloc[0] for _, val in self.mappings.get(task).iterrows()}
            for task in self.tasks
        }

    def __getitem__(self, index) -> dict:
        image_name = self.dataset.iloc[index, 0]
        tasks = self.dataset.iloc[index, 1:]
        image = self._load_image(image_fname=image_name)
        image = self.preprocess(image) if self.preprocess else image

        gts = {task: self.mappings_dict[task][tasks[task]] for task in self.tasks}

        return {DataDict.IMAGE: image, DataDict.GTS: gts}

    def __len__(self) -> int:
        return len(self.dataset)

    def init_dataset(self, dataset, **kwargs):
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

    def _load_image(self, image_fname) -> Image.Image:
        return Image.open(f"{self.img_dir}/{image_fname}").convert("RGB")
