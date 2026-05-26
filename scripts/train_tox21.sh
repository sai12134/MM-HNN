#!/bin/bash

ANACONDA_PYTHON_PATH="C:\Users\16122\miniconda3\envs\mhnn\python.exe"

$ANACONDA_PYTHON_PATH -u train_tox21.py \
  --data_dir datasets \
  --delocalization \
  --runs 3 \
  --seed 42 \
  --device 0 \
  --epochs 300 \
  --batch_size 32 \
  --lr 0.000334 \
  --min_lr 0.0001 \
  --wd 0.000677 \
  --log_steps 1 \
  --method mhnn \
  --All_num_layers 2 \
  --MLP1_num_layers 2 \
  --MLP2_num_layers 1 \
  --MLP3_num_layers 1 \
  --MLP4_num_layers 2 \
  --MLP_hidden 256 \
  --output_num_layers 3 \
  --output_hidden 162 \
  --aggregate mean \
  --normalization ln \
  --activation prelu \
  --dropout 0.047298