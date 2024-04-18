from torch.utils.data import Dataset
from typing import Any

class ImgeGraphDataset(Dataset):
    # need of a class similar to image-text-dataset
    # each data point is img, cls
    # each batch should be images, styles to take, gts
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    
    def __len__(self):
        return 0
    
    def collate_fn(self, batch):
        pass



