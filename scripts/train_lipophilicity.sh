#!/bin/bash

ANACONDA_PYTHON_PATH="C:\Users\16122\miniconda3\envs\mhnn\python.exe"

$ANACONDA_PYTHON_PATH -u train.py \
  --data_dir datasets \
  --dataset lipophilicity \
  --delocalization \
  --runs 10 \
  --seed 42 \
  --device 0 \
  --epochs 300 \
  --batch_size 32 \
  --lr 0.000272 \
  --min_lr 0.0001 \
  --wd 0.000635 \
  --log_steps 1 \
  --method mhnn \
  --All_num_layers 2 \
  --MLP1_num_layers 2 \
  --MLP2_num_layers 3 \
  --MLP3_num_layers 1 \
  --MLP4_num_layers 2 \
  --MLP_hidden 256 \
  --output_num_layers 2 \
  --output_hidden 130 \
  --aggregate mean \
  --normalization ln \
  --activation relu \
  --dropout 0.110676