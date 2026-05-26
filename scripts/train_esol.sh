#!/bin/bash

ANACONDA_PYTHON_PATH="C:\Users\16122\miniconda3\envs\mhnn\python.exe"

$ANACONDA_PYTHON_PATH -u train.py \
  --data_dir datasets \
  --dataset esol \
  --delocalization \
  --runs 10 \
  --seed 42 \
  --device 0 \
  --epochs 300 \
  --batch_size 32 \
  --lr 0.00013 \
  --min_lr 0.0001 \
  --wd 0.00039 \
  --log_steps 1 \
  --method mhnn \
  --All_num_layers 3 \
  --MLP1_num_layers 2 \
  --MLP2_num_layers 2 \
  --MLP3_num_layers 1 \
  --MLP4_num_layers 2 \
  --MLP_hidden 128 \
  --output_num_layers 2 \
  --output_hidden 194 \
  --aggregate mean \
  --normalization ln \
  --activation relu \
  --dropout 0.148