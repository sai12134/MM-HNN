import pandas as pd
import numpy as np
import deepchem as dc
import os
import argparse


def random_split(file_path, train_ratio=0.8, val_ratio=0.1):
    print('random split')
    np.random.seed(2024)
    df = pd.read_csv(file_path)
    shuffled_df = df.sample(frac=1)
    total_size = len(df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_df = shuffled_df.iloc[:train_size]
    val_df = shuffled_df.iloc[train_size:train_size + val_size]
    test_df = shuffled_df.iloc[train_size + val_size:]

    train_filename = os.path.join('datasets/raw', "smiles_train.csv")
    val_filename = os.path.join('datasets/raw', "smiles_valid.csv")
    test_filename = os.path.join('datasets/raw', "smiles_test.csv")

    train_df.to_csv(train_filename, index=False)
    val_df.to_csv(val_filename, index=False)
    test_df.to_csv(test_filename, index=False)


def scaffold_split(file_path):
    print('scaffold split')
    df = pd.read_csv(file_path)
    smiles = df.iloc[:, 0].astype(str).values
    labels = df.iloc[:, 1].values

    dataset = dc.data.NumpyDataset(X=smiles, y=labels, ids=smiles)
    scaffold_splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = scaffold_splitter.train_valid_test_split(dataset)

    def dataset_to_df(dataset):
        smiles = dataset.ids
        labels = dataset.y
        return pd.DataFrame({'smiles': smiles, 'lable': labels})

    train_df = dataset_to_df(train_dataset)
    train_df.to_csv('datasets/raw/smiles_train.csv', index=False)

    valid_df = dataset_to_df(valid_dataset)
    valid_df.to_csv('datasets/raw/smiles_valid.csv', index=False)

    test_df = dataset_to_df(test_dataset)
    test_df.to_csv('datasets/raw/smiles_test.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dataset splitting')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'bbbp', 'tox21'])
    args = parser.parse_args()
    print(args)

    dataset = args.dataset

    file_path = f'datasets/raw/{dataset}.csv'
    if dataset in ['esol', 'freeSolv', 'lipophilicity', 'tox21']:
        random_split(file_path)
    else:
        scaffold_split(file_path)
