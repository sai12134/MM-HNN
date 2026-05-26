#!/bin/bash

ANACONDA_PYTHON_PATH="C:\Users\16122\miniconda3\envs\mhnn\python.exe"

$ANACONDA_PYTHON_PATH -u train.py \
  --data_dir datasets \
  --dataset hiv \
  --delocalization \
  --runs 10 \
  --seed 42 \
  --device 0 \
  --epochs 300 \
  --batch_size 32 \
  --lr 0.00013 \
  --min_lr 0.0001 \
  --wd 0.000924 \
  --log_steps 1 \
  --method mhnn \
  --All_num_layers 2 \
  --MLP1_num_layers 3 \
  --MLP2_num_layers 1 \
  --MLP3_num_layers 1 \
  --MLP4_num_layers 2 \
  --MLP_hidden 256 \
  --output_num_layers 3 \
  --output_hidden 158 \
  --aggregate mean \
  --normalization ln \
  --activation relu \
  --dropout 0.054007