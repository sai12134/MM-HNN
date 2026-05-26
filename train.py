import time
import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from models import MHNN, MHNNS
from datasets import HGraph, OneTarget
from utils import Logger, seed_everything
from split import random_split, scaffold_split
import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve

import os
import shutil


@torch.no_grad()
def evaluate_reg(args, model, loader, std=None):
    model.eval()
    err_squared_sum = 0.0
    # for RMSE
    for batch in loader:
        batch = batch.to(args.device)
        out = model(batch)
        if std is not None:
            diff = (out * std - batch.y * std)
            err_squared_sum += diff.pow(2).sum().item()
        else:
            diff = (out - batch.y)
            err_squared_sum += diff.pow(2).sum().item()
    mse = err_squared_sum / len(loader.dataset)
    rmse = mse ** 0.5
    return rmse


@torch.no_grad()
def evaluate_cls(args, model, loader):
    model.eval()
    all_scores = []
    all_labels = []
    device = args.device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model(batch).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = trapezoid(tpr, fpr)
    return roc_auc


if __name__ == '__main__':

    print('Task start time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Training')

    # 数据集参数
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, help='[esol, freesolv,lipophilicity, hiv, bace, bbbp]')

    # 训练超参数
    parser.add_argument('--delocalization', action='store_true', help='includes the delocalization hyperedges')
    parser.add_argument('--runs', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--min_lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--clip_gnorm', default=None, type=float)
    parser.add_argument('--log_steps', type=int, default=1)

    # 模型超参数
    parser.add_argument('--method', default='mhnn', help='model type')
    parser.add_argument('--All_num_layers', default=3, type=int, help='number of basic blocks')
    parser.add_argument('--MLP1_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP2_num_layers', default=2, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=2, type=int, help='layer number of mlp3')
    parser.add_argument('--MLP4_num_layers', default=2, type=int, help='layer number of mlp4')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--output_num_layers', default=2, type=int)
    parser.add_argument('--output_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'])
    parser.add_argument('--activation', default='relu', choices=['Id', 'relu', 'prelu'])
    parser.add_argument('--dropout', default=0.0, type=float)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset in ['esol', 'freesolv', 'lipophilicity']:
        task_type = 'reg'
    else:
        task_type = 'cls'

    file_path = f'datasets/raw/{args.dataset}.csv'
    if args.dataset in ['hiv', 'bace', 'bbbp']:
        scaffold_split(file_path)
    else:
        random_split(file_path)

    processed_dir = 'datasets/processed'

    if os.path.exists(processed_dir):
        print(f'\nRemoving old processed data: {processed_dir}')
        shutil.rmtree(processed_dir)

    os.makedirs(processed_dir, exist_ok=True)

    transform = T.Compose([OneTarget()])

    train_dataset = HGraph(root=args.data_dir, partition='train', transform=transform,
                           delocalization=args.delocalization)
    valid_dataset = HGraph(root=args.data_dir, partition='valid', transform=transform,
                           delocalization=args.delocalization)
    test_dataset = HGraph(root=args.data_dir, partition='test', transform=transform, delocalization=args.delocalization)

    if task_type == 'reg':
        mean = train_dataset.data.y.mean(dim=0, keepdim=True)
        std = train_dataset.data.y.std(dim=0, keepdim=True)

        train_dataset.data.y = (train_dataset.data.y - mean) / std
        valid_dataset.data.y = (valid_dataset.data.y - mean) / std
        test_dataset.data.y = (test_dataset.data.y - mean) / std

        mean, std = mean[:, 0].item(), std[:, 0].item()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化一个日志记录器，用于记录多次运行的实验结果
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed = args.seed
        seed_everything(seed=seed, workers=True)
        print(f'\nRun No. {run+1}:')
        print(f'Seed: {seed}\n')

        model = MHNNS(1, args, task_type)
        model = model.to(device)
        print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        if task_type == 'reg':
            loss_fn = nn.MSELoss()
            best_val_RMSE = None
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                                   patience=5, min_lr=args.min_lr)
        else:
            loss_fn = nn.BCELoss()
            best_val_AUC = None
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7,
                                                                   patience=5, min_lr=args.min_lr)

        # training
        if task_type == 'reg':
            for epoch in range(1, 1 + args.epochs):
                model.train()
                loss_all = 0.0
                lr = scheduler.optimizer.param_groups[0]['lr']
                for data in train_loader:
                    data = data.to(args.device)
                    optimizer.zero_grad()
                    out = model(data)
                    loss = loss_fn(out, data.y)
                    loss.backward()
                    loss_all += loss.item() * data.num_graphs
                    if args.clip_gnorm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gnorm)
                    optimizer.step()

                loss_all /= len(train_loader.dataset)
                valid_RMSE = evaluate_reg(args, model, valid_loader, std=std)
                scheduler.step(valid_RMSE)
                if best_val_RMSE is None or valid_RMSE < best_val_RMSE:
                    test_RMSE = evaluate_reg(args, model, test_loader, std=std)
                    best_val_RMSE = valid_RMSE
                logger.add_result(run, [loss_all, valid_RMSE, test_RMSE])

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'lr: {lr:.6f}, '
                          f'Loss: {loss_all:.6f}, '
                          f'Valid RMSE: {valid_RMSE:.6f}, '
                          f'Test RMSE: {test_RMSE:.6f}')
        else:
            for epoch in range(1, 1 + args.epochs):
                model.train()
                loss_all = 0.0
                lr = scheduler.optimizer.param_groups[0]['lr']
                for data in train_loader:
                    data = data.to(args.device)
                    optimizer.zero_grad()
                    out = model(data)
                    loss = loss_fn(out, data.y.view(-1).float())
                    loss.backward()
                    loss_all += loss.item() * data.num_graphs
                    if args.clip_gnorm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gnorm)
                    optimizer.step()

                loss_all /= len(train_loader.dataset)
                valid_AUC = evaluate_cls(args, model, valid_loader)
                scheduler.step(valid_AUC)
                if best_val_AUC is None or valid_AUC > best_val_AUC:
                    test_AUC = evaluate_cls(args, model, test_loader)
                    best_val_AUC = valid_AUC
                logger.add_result(run, [loss_all, valid_AUC, test_AUC])

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'lr: {lr:.6f}, '
                          f'Loss: {loss_all:.6f}, '
                          f'Valid AUC: {valid_AUC:.6f}, '
                          f'Test AUC: {test_AUC:.6f}')

        logger.print_statistics(run=run, task_type=task_type)
    logger.print_statistics(run=None, task_type=task_type)

print('Task end time:')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
end_time = time.time()
print('Total time taken: {} s.'.format(int(end_time - start_time)))
