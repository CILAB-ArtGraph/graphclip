import os
import open_clip
import pandas as pd
import torch
from src.utils import load_ruamel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from safetensors.torch import save_file


class LabelDataset(Dataset):
    def __init__(self, data, tokenizer, target_col=1, **in_kwargs):
        self.data = pd.read_csv(data, **in_kwargs) if isinstance(data, str) else data
        self.tokenizer = tokenizer
        self.target_col = target_col

    def __getitem__(self, item):
        name = self.data.iloc[item, self.target_col]
        return self.tokenizer(name).squeeze()

    def __len__(self):
        return len(self.data)


def get_labels(in_dir, exceptions, mapping):
    return map(
        lambda x: (mapping.get(x[0], x[0]), x[1]),
        filter(
            lambda x: x[0] not in exceptions,
            map(lambda x: (x.split("_")[0], x), os.listdir(in_dir)),
        ),
    )


@torch.no_grad()
def extract_label_clip(parameters: dict):
    device = parameters.get("device")
    model, _, _ = open_clip.create_model_and_transforms(**parameters.get("model"))
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(parameters["tokenizer"])
    labels = get_labels(**parameters.get("labels"))
    external_source = parameters.get("source", {})
    out_dir = parameters.get("out_dir")
    os.makedirs(out_dir, exist_ok=True)
    root_dir = parameters.get("labels").get("in_dir")
    for lab, filename in labels:
        print(f"Extracting textual features for {lab}...")
        target_df = pd.read_csv(f'{root_dir}/{filename}', header=None)
        lab_source_fname = external_source.get(lab, None)
        target_col=1
        if lab_source_fname is not None:
            lab_source = pd.read_csv(lab_source_fname)
            target_df = pd.merge(left=target_df, right=lab_source, left_on=1, right_on="style")
            target_col=2
        dataset = LabelDataset(
            data=target_df,
            tokenizer=tokenizer,
            target_col=target_col,
        )
        loader = DataLoader(dataset, **parameters["dataset"])
        tensors = None
        for tokens in tqdm(loader):
            tokens = tokens.to(device)
            embeddings = model.encode_text(tokens)
            tensors = (
                embeddings if tensors is None else torch.cat([tensors, embeddings])
            )
        save_file({"embeddings": tensors}, f"{out_dir}/{lab}.safetensors")
        print("Done!")
