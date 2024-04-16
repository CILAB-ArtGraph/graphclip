from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from tqdm import tqdm
from safetensors.torch import save_file
from open_clip import CLIP
from open_clip import create_model_and_transforms, get_tokenizer


class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess=None):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.images = os.listdir(image_dir)

    def __getitem__(self, item):
        img_name = self.images[item]
        image = Image.open(f"{self.image_dir}/{img_name}").convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        return img_name[:-4], image

    def __len__(self):
        return len(self.images)



def loop(model: CLIP, dataloader: DataLoader, out_dir: str, device: str):
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    for names, images in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
        for name, image in zip(names, image_embeddings):
            data = {"image": image.squeeze().contiguous()}
            save_file(data, f"{out_dir}/{name}.safetensors")

@torch.no_grad
def extract_vis_clip(parameters: dict):
    model, _, preprocess = create_model_and_transforms(**parameters.get("model"))
    data_params = parameters.get("dataset")
    data_params.update({"preprocess": preprocess})
    dataset = ImageDataset(**data_params)
    dataloader = DataLoader(dataset=dataset, **parameters.get("dataloader"))
    loop(model=model, dataloader=dataloader, **parameters.get("loop"))
