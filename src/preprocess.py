from sklearn.model_selection import train_test_split
import pandas as pd
import os


def preprocess_normal(
    dataset_path: str,
    label_column: str,
    val_size: float,
    test_size: float,
    outdir: str,
    stratification: bool=True,
    random_state: int = None,
    out_args: dict = {},
    **kwargs,
):
    dataset = pd.read_csv(dataset_path, **kwargs)
    val_num = int(len(dataset) * val_size)
    test_num = int(len(dataset) * test_size)
    
    
    trainval, test = train_test_split(dataset, test_size=test_num, random_state=random_state, stratify=dataset[label_column] if stratification else None)
    train, val = train_test_split(trainval, test_size=val_num, random_state=random_state, stratify=trainval[label_column] if stratification else None)
    
    # SAVE
    os.makedirs(outdir, exist_ok=True)
    train.to_csv(f"{outdir}/train.csv", **out_args)
    val.to_csv(f"{outdir}/val.csv", **out_args)
    test.to_csv(f"{outdir}/test.csv", **out_args)