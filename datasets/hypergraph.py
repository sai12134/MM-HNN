import os
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from datasets.utils import smi2hgraph, HData, edge_order


class HGraph(InMemoryDataset):
    def __init__(self, root, partition='train', transform=None, pre_transform=None, pre_filter=None,
                 delocalization=None, task=None):
        assert partition in ['train', 'valid', 'test']
        self.partition = partition
        self.delocalization = delocalization
        self.task = task

        super().__init__(root, transform, pre_transform, pre_filter)

        if self.partition == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.partition == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])
        self.ids = self.data.smi

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_file_names(self):
        return ['smiles_train.csv', 'smiles_valid.csv', 'smiles_test.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def compute_hgraph_data(self, df):
        if self.task is None:
            smiles = df['smiles'].values.tolist()
            target = df.iloc[:, 1:].values
        else:
            df = df.dropna(subset=[df.columns[self.task]])
            smiles = df['smiles'].values.tolist()
            target = df.iloc[:, [self.task]].values
        target = torch.tensor(target, dtype=torch.float)

        data_list = []
        # 遍历字符串列表
        for i, smi in enumerate(tqdm(smiles)):

            atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi, delocalization=self.delocalization)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = target[i].unsqueeze(0)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(x=x, y=y, n_e=n_e, smi=smi,
                         edge_index0=edge_index0,
                         edge_index1=edge_index1,
                         edge_attr=edge_attr,
                         e_order=e_order)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        for path in self.raw_paths:
            filename = os.path.basename(path)
            if "smiles_train" in filename:
                df = pd.read_csv(path)
                data_list = self.compute_hgraph_data(df)
                torch.save(self.collate(data_list), self.processed_paths[0])
            elif "smiles_valid" in filename:
                df = pd.read_csv(path)
                data_list = self.compute_hgraph_data(df)
                torch.save(self.collate(data_list), self.processed_paths[1])
            elif "smiles_test" in filename:
                df = pd.read_csv(path)
                data_list = self.compute_hgraph_data(df)
                torch.save(self.collate(data_list), self.processed_paths[2])
