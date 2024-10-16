import torch_geometric as pyg
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from random import shuffle
import torch_geometric.data
from typing import Union, Optional
from src.utils import StrEnum
from torch_geometric.transforms import ToUndirected, AddSelfLoops
from sklearn.model_selection import train_test_split


class MappingKeys(StrEnum):
    IDX = "idx"
    NAME = "name"
    ARTWORK = "artwork"
    STYLE = "style"


class ArtGraphInductiveSplitter:
    def __init__(
        self,
        dataset: pyg.data.HeteroData,
        node_split_center: str,
        stratify=None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed=None,
    ):
        self.dataset = dataset
        self.node_split_center = node_split_center
        self.stratify = stratify
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio = 1 - (self.val_ratio + self.test_ratio)
        self.loop_nodes = []
        self.seed = seed
        if self.seed:
            pyg.seed_everything(self.seed)

    def __stratify_split(self):
        edge_list = self.dataset[
            self.node_split_center, self.stratify
        ].edge_index.T.contiguous()
        train_size = int(edge_list.size(0) * self.train_ratio)
        val_size = int(edge_list.size(0) * self.val_ratio)
        train_edges, drop_edges = train_test_split(
            edge_list,
            train_size=train_size,
            random_state=self.seed,
            stratify=edge_list[:, 1],
        )

        val_edges, test_edges = train_test_split(
            drop_edges,
            train_size=val_size,
            random_state=self.seed,
            stratify=drop_edges[:, 1],
        )
        train_central_nodes = train_edges[:, 0]
        val_central_nodes = val_edges[:, 0]
        test_central_nodes = test_edges[:, 0]
        assert train_central_nodes.size(0) + val_central_nodes.size(
            0
        ) + test_central_nodes.size(0) == edge_list.size(0)
        return train_central_nodes, val_central_nodes, test_central_nodes

    def __random_split(self):
        permutation = torch.tensor(
            shuffle(list(range(self.dataset[self.node_split_center].x.size(0))))
        )
        train_size = int(permutation.size(0) * self.train_ratio)
        val_size = int(permutation.size(0) * self.val_ratio)
        train_central_nodes = permutation[:train_size]
        val_central_nodes = permutation[train_size : train_size + val_size]
        test_central_nodes = permutation[train_size + val_size :]
        assert train_central_nodes.size(0) + val_central_nodes.size(
            0
        ) + test_central_nodes.size(0) == permutation.size(0)
        return train_central_nodes, val_central_nodes, test_central_nodes

    def get_split(self):
        if not self.stratify:
            return self.__random_split()
        return self.__stratify_split()

    def transform_split(self, central_nodes):
        data = deepcopy(self.dataset)
        ids = {self.node_split_center: central_nodes.tolist()}

        central_edge_types = filter(
            lambda x: x[0] == self.node_split_center, data.metadata()[1]
        )
        for _, (source, _, dest) in tqdm(enumerate(central_edge_types)):
            # take edge list
            edge_list = pd.DataFrame(data[(source, _, dest)].edge_index.T.cpu().numpy())
            # filter for central nodes
            edge_list = edge_list[edge_list[0].isin(central_nodes.tolist())]
            # set edge list
            data[(source, _, dest)].edge_index = torch.from_numpy(
                edge_list.values
            ).T.contiguous()
            # save original ids
            ids[dest] = edge_list[1].unique().tolist()

        non_central_edge_types = filter(
            lambda x: x[0] != self.node_split_center, data.metadata()[1]
        )
        for _, (source, _, dest) in tqdm(enumerate(non_central_edge_types)):
            assert source in ids, f"{source} not in {list(ids.keys())}"
            # take edge list
            edge_list = pd.DataFrame(data[(source, _, dest)].edge_index.T.cpu().numpy())
            # get nodes already saved (for the second hop)
            source_nodes = ids[source]
            # filter edge list
            edge_list = edge_list[edge_list[0].isin(source_nodes)]
            # set edge list
            data[(source, _, dest)].edge_index = torch.from_numpy(
                edge_list.values
            ).T.contiguous()
            # save original ids
            ids[dest] = list(set(ids.get(dest, []) + edge_list[1].unique().tolist()))

        map_ids = {
            t: {old_id: new_id for old_id, new_id in zip(ids[t], range(len(ids[t])))}
            for t in ids.keys()
        }
        # reset relationships
        for source, _, dest in data.metadata()[1]:
            edge_list = pd.DataFrame(data[(source, _, dest)].edge_index.T.cpu().numpy())
            edge_list[0] = edge_list[0].map(map_ids[source])
            edge_list[1] = edge_list[1].map(map_ids[dest])
            data[(source, _, dest)].edge_index = (
                torch.from_numpy(edge_list.values).T.contiguous().type(torch.LongTensor)
            )
            # additional check
            difference = set(edge_list[0].unique().tolist()).difference(
                set(list(range(data[source].x.size(0))))
            )
            if len(difference) != 0:
                print(f"cannot recognize {source} number {difference}")

            difference = set(edge_list[1].unique().tolist()).difference(
                set(list(range(data[dest].x.size(0))))
            )
            if len(difference) != 0:
                print(f"cannot recognize {dest} number {difference}")

        # reset features
        for node_type in data.metadata()[0]:
            data[node_type].x = data[node_type].x[ids[node_type]]
        return data, map_ids

    def transform(self):
        train_central_nodes, val_central_nodes, test_central_nodes = self.get_split()
        print("Making training split...")
        train_data, train_map = self.transform_split(train_central_nodes)
        print("Done!")
        print("Making validation split...")
        val_data, val_map = self.transform_split(val_central_nodes)
        print("Done!")
        print("Making test split...")
        test_data, test_map = self.transform_split(test_central_nodes)
        print("Done!")
        return (train_data, train_map), (val_data, val_map), (test_data, test_map)


class ArtGraphInductivePruner:
    """Given a HeteroData object and the artwork to mantain, it discards all the others"""

    def __init__(
        self,
        data: torch_geometric.data.HeteroData,
        mapping: Union[str, pd.DataFrame],
        source: Union[str, pd.DataFrame],
        mapping_kwargs={},
        source_kwargs={},
        add_self_loops: bool = True,
        to_undirected: bool = True,
        sample: Optional[float] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.data = data
        self.mapping = self._load_data(mapping, **mapping_kwargs)
        self.source = self._load_data(source, **source_kwargs)
        self.add_self_loops = add_self_loops
        self.to_undirected = to_undirected
        self.sample = sample
        self.random_state = random_state

    def _get_pruned_artworks(self) -> dict[int, int]:
        pruned_artworks = pd.merge(
            left=self.source,
            right=self.mapping,
            left_on=MappingKeys.ARTWORK,
            right_on=MappingKeys.NAME,
        )
        if self.sample is not None:
            pruned_artworks = train_test_split(
                pruned_artworks,
                test_size=self.sample,
                stratify=pruned_artworks[MappingKeys.STYLE],
                random_state=self.random_state,
            )[1]
        pruned_artworks = pruned_artworks.sort_values(by=MappingKeys.IDX)[
            MappingKeys.IDX
        ].tolist()
        return {old_idx: new_idx for (new_idx, old_idx) in enumerate(pruned_artworks)}

    def transform(self) -> torch_geometric.data.HeteroData:
        out_data = deepcopy(self.data)
        artwork_mapping = self._get_pruned_artworks()
        out_data["artwork"].x = out_data["artwork"].x[list(artwork_mapping.keys())]
        for etype in self.data.metadata()[
            1
        ]:  # for each edge type transform the edge index
            (h, _, t) = etype
            if h == "artwork":
                target_col = 0
            elif t == "artwork":
                target_col = 1
            else:
                continue
            edge_index = pd.DataFrame(out_data[etype].edge_index.T.numpy())
            # filtering
            edge_index = edge_index[
                edge_index[target_col].isin(list(artwork_mapping.keys()))
            ]
            # mapping
            edge_index[target_col] = edge_index[target_col].map(artwork_mapping)
            out_data[etype].edge_index = torch.from_numpy(edge_index.values).long().T

        # postprocess
        if self.add_self_loops:
            out_data = AddSelfLoops()(out_data)
        if self.to_undirected:
            out_data = ToUndirected()(out_data)

        return out_data, artwork_mapping

    def _load_data(self, data: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        return data if isinstance(data, pd.DataFrame) else pd.read_csv(data, **kwargs)


class ArtGraphInductiveClassPruner:
    def __init__(
        self,
        data: torch_geometric.data.HeteroData,
        num_classes: dict[str, int],
        mapping_classes: dict[str, Union[str, pd.DataFrame]],
        mapping_kwargs: dict = {},
        random_state=None,
        add_self_loops: bool = True,
        to_undirected: bool = True,
    ) -> None:
        self.data = data
        self.num_classes = num_classes
        self.mapping_classes = {
            k: self._load_data(v, **mapping_kwargs) for k, v in mapping_classes.items()
        }
        self.random_state = random_state
        self.add_self_loops = add_self_loops
        self.to_undirected = to_undirected

    def _load_data(self, data: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        return data if isinstance(data, pd.DataFrame) else pd.read_csv(data, **kwargs)

    def _get_pruned_classes(self) -> dict[str, dict[int, int]]:
        # style -> idx, name
        pruned_classes = {
            k: train_test_split(
                v, test_size=self.num_classes[k], random_state=self.random_state
            )[1]
            for k, v in self.mapping_classes.items()
        }
        new_mapping = {
            k: {n: o for o, n in enumerate(v["idx"].unique().tolist())}
            for k, v in pruned_classes.items()
        }
        return pruned_classes, new_mapping

    def transform(self):
        out_data = deepcopy(self.data)
        classes, mapping = self._get_pruned_classes()

        # reassign node features
        for cls in classes:
            out_data[cls].x = out_data[cls].x[list(mapping[cls].keys())]

            for etype in self.data.metadata()[
                1
            ]:  # for each edge type transform the edge index
                (h, _, t) = etype
                if h == cls:
                    target_col = 0
                elif t == cls:
                    target_col = 1
                else:
                    continue
                edge_index = pd.DataFrame(out_data[etype].edge_index.T.numpy())
                # filtering
                edge_index = edge_index[
                    edge_index[target_col].isin(list(mapping[cls].keys()))
                ]
                # mapping
                edge_index[target_col] = edge_index[target_col].map(mapping[cls])
                out_data[etype].edge_index = (
                    torch.from_numpy(edge_index.values).long().T
                )
        # postprocess
        if self.add_self_loops:
            out_data = AddSelfLoops()(out_data)
        if self.to_undirected:
            out_data = ToUndirected()(out_data)

        # postprocessing classes
        for cls, df in classes.items():
            df["idx"] = df["idx"].map(mapping[cls])
            df.reset_index(drop=True, inplace=True)
            classes[cls] = df
        return out_data, classes
