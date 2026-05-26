# MM-HNN
# MM-HNN: A MoleculeNet-based Molecular Hypergraph Neural Network for Molecular Property Prediction

## 🚀 Requirements
```
python                    3.9
torch                     2.3.1
torch-geometric           2.5.3
torch-scatter             2.1.2
rdkit                     2024.03.3
deepchem                  2.6.0
ogb                       1.3.6
numpy                     1.25.2
pandas                    2.2.2
scipy                     1.13.1
scikit-learn              1.6.1
tqdm                      4.66.4
```

## 📌 Datasets

| Dataset       | Compounds |    Task type   | Task number |  Metric |
|:-------------:|:---------:|:--------------:|:-----------:|:-------:|
| ESOL          | 1,128     |   regression   | 1           | RMSE    |
| FreeSolv      | 642       |   regression   | 1           | RMSE    |
| Lipophilicity | 4,200     |   regression   | 1           | RMSE    |
| HIV           | 41,127    | classification | 1           | ROC-AUC |
| BACE          | 1,513     | classification | 1           | ROC-AUC |
| BBBP          | 2,039     | classification | 1           | ROC-AUC |
| Tox21         | 7,831     | classification | 12          | ROC-AUC |

All the original datasets can be obtained from https://moleculenet.org/datasets-1.

## 🔥 Model Training
You can train the model by
```
bash scripts/train_[dataset_name].sh
```
