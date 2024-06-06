from .db_manager import DBManager, get_stat_queries
from .raw_generation_nosplit import ArtGraphNoSplit
from sklearn.model_selection import train_test_split
from .graph_splitter import ArtGraphInductivePruner, ArtGraphInductiveClassPruner
from .load_artgraph import ArtGraph
import pandas as pd
import os
import torch
import json
from copy import deepcopy


def download_artgraph(parameters):
    db_manager_params = {
        k: v for k, v in parameters["connection"].items() if k != "database"
    }
    db_manager = DBManager(**db_manager_params)
    db = parameters["connection"]["database"]
    queries = {
        "mapping": db_manager.get_mapping_queries(db),
        "relations": db_manager.get_relation_queries(db),
        "stats": get_stat_queries(),
    }
    artgraph = ArtGraphNoSplit(
        root=parameters["out"],
        conf=parameters["connection"],
        queries=queries,
        artwork_subset=db_manager.get_artworks(db),
    )
    artgraph.build()
    artgraph.write()


def split_normal(
    dataset_path: str,
    val_size: float,
    test_size: float,
    outdir: str,
    stratification: bool = True,
    random_state: int = None,
    columns_to_drop: list[str] = None,
    label_column: str = None,
    out_args: dict = {},
    **kwargs,
):
    dataset = pd.read_csv(dataset_path, **kwargs)
    if columns_to_drop is not None:
        dataset = dataset.drop(columns_to_drop, axis=1)
    val_num = int(len(dataset) * val_size)
    test_num = int(len(dataset) * test_size)

    trainval, test = train_test_split(
        dataset,
        test_size=test_num,
        random_state=random_state,
        stratify=dataset[label_column] if stratification else None,
    )
    train, val = train_test_split(
        trainval,
        test_size=val_num,
        random_state=random_state,
        stratify=trainval[label_column] if stratification else None,
    )

    # SAVE
    os.makedirs(outdir, exist_ok=True)
    train.to_csv(f"{outdir}/train.csv", **out_args)
    val.to_csv(f"{outdir}/val.csv", **out_args)
    test.to_csv(f"{outdir}/test.csv", **out_args)


def split_normal_classes(
    in_dir: str,
    mapping_dir: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    for file in filter(lambda x: x != "test.csv", os.listdir(in_dir)):
        df = pd.read_csv(f"{in_dir}/{file}")
        target_cols = list(filter(lambda x: x != "artwork", df.columns))
        mappings = {
            col: pd.read_csv(f"{mapping_dir}/{col}_mapping.csv")
            for col in target_cols
        }
        out_df = deepcopy(df)
        for col in target_cols:
            df = df[df[col].isin(mappings[col]["name"].unique().tolist())]
        df.reset_index(drop=True, inplace=True)
        df.to_csv(f'{out_dir}/{file}', index=False)


def split_graph(parameters: dict):
    data_params = parameters.get("data")
    data = ArtGraph(**data_params)[0]
    out_data, artwork_mapping = ArtGraphInductivePruner(
        data=data, **parameters.get("pruner")
    ).transform()
    out_dir = parameters.get("out_dir")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(out_data, f"{out_dir}/train_graph.pt")
    with open(f"{out_dir}/artwork_mapping.json", "w+") as f:
        json.dump(artwork_mapping, f)


def split_classes(parameters: dict):
    data_params = parameters.get("data")
    data = torch.load(**data_params)
    out_data, mapping = ArtGraphInductiveClassPruner(
        data=data, **parameters.get("pruner")
    ).transform()
    out_dir = parameters.get("out_dir")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(out_data, f"{out_dir}/train_graph.pt")
    for k, v in mapping.items():
        v.to_csv(f"{out_dir}/{k}_mapping.csv", index=False)


def split(parameters: dict, graph: bool = False, classes: bool = False):
    if not graph:
        if classes:
            split_normal_classes(**parameters)
        else:
            split_normal(**parameters)
    elif classes:
        split_classes(parameters=parameters)
    else:
        split_graph(parameters=parameters)
