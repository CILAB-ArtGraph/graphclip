from enum import IntEnum
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import os
from torch_geometric.data import InMemoryDataset
import src.utils as utils


class LabelEncoder(IntEnum):
    SCALAR = 0
    ONE_HOT = 1
    OPEN_CLIP = 2


class VisFeatEncoder(IntEnum):
    CLIP = 0
    OPEN_CLIP = 1
    TIMM_VIT = 2
    RESNET = 3


class FileMapping(utils.StrEnum):
    MAPPING = 'mapping'
    RAW = 'raw'
    MAPPING_SUFFIX = '_entidx2name'
    RELATIONS = 'relations'


class ArtGraph(InMemoryDataset):
    def __init__(
            self, root,
            vis_feats_root,
            label_feats_root=None,
            labels=LabelEncoder.OPEN_CLIP,
            vis_feats=VisFeatEncoder.OPEN_CLIP,
    ):
        self.labels = labels
        self.vis_feats = vis_feats
        self.root = root
        self.label_feats_root = label_feats_root
        self.vis_feats_root = vis_feats_root

        assert self.labels in list(LabelEncoder)
        assert self.vis_feats in list(VisFeatEncoder)
        super().__init__(root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        for f in os.listdir(fr'{root}/processed'):
            os.remove(fr'{root}/processed/{f}')
        os.rmdir(fr'{root}/processed')

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv'
        ]

        return file_names

    @property
    def processed_file_names(self):
        return 'none.pt'

    def download(self):
        return

    def __get_artwork_features(self):
        features = torch.empty(size=(116475, 512))
        mapping = pd.read_csv(f'{self.root}/{FileMapping.MAPPING.value}/artwork{FileMapping.MAPPING_SUFFIX.value}.csv',
                              header=None)
        for ix, name in mapping.values:
            name = name[:-4] + '.safetensors'
            features[ix] = utils.load_tensor(f'{self.vis_feats_root}/{name}', key='image')
        return features

    def __get_node_types(self, exceptions: dict):
        base = list(
            filter(
                lambda x: x != 'artwork', map(
                    lambda x: x[:-len(FileMapping.MAPPING_SUFFIX.value)],
                    os.listdir(f'{self.root}/{FileMapping.MAPPING.value}')
                )
            )
        )
        return list(map(lambda x: exceptions.get(x, x).split('_')[0], base))

    def process(self):
        data = pyg.data.HeteroData()

        # get artwork visual features
        data['artwork'].x = self.__get_artwork_features()

        path = os.path.join(self.raw_dir, 'num-node-dict.csv')
        num_nodes_df = pd.read_csv(path)
        exceptions = {"training": "training_node"}
        num_nodes_df.rename(columns=exceptions, inplace=True)
        nodes_type = self.__get_node_types(exceptions)  # add map training nod

        # get label features
        if self.labels == LabelEncoder.SCALAR:
            for feature, node_type in enumerate(filter(lambda x: x != 'artwork', num_nodes_df.columns)):
                ones = [feature + 1] * num_nodes_df[node_type].tolist()[0]
                data_tensor = torch.tensor(ones)
                data_tensor = torch.reshape(data_tensor, (num_nodes_df[node_type].tolist()[0], 1))
                data[node_type].x = data_tensor.type(torch.FloatTensor)
        elif self.labels == LabelEncoder.ONE_HOT:
            for node_type in filter(lambda x: x != 'artwork', num_nodes_df.columns):
                data[node_type].x = torch.eye(num_nodes_df[node_type].tolist()[0])
        elif self.labels == LabelEncoder.OPEN_CLIP:
            for nodes_type in filter(lambda x: x != 'artwork', num_nodes_df.columns):
                data[nodes_type].x = utils.load_tensor(file=f'{self.label_feats_root}/{nodes_type}.safetensors',
                                                       key='embeddings',
                                                       framework='pt',
                                                       device='cpu')

        # add edges
        for edge_type in os.listdir(fr'{self.raw_dir}/relations'):
            sub, verb, obj = edge_type.split("___")
            path = fr'{self.raw_dir}/relations/{edge_type}/edge.csv'
            edge_index = pd.read_csv(path, header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous().type(torch.LongTensor)
            if obj == 'training':
                obj = 'training_node'
            data[(sub, verb, obj)].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_features(self):
        return self.data['artist'].x.shape[1]

