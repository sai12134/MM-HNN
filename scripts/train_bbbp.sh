#!/bin/bash

ANACONDA_PYTHON_PATH="C:\Users\16122\miniconda3\envs\mhnn\python.exe"

$ANACONDA_PYTHON_PATH -u train.py \
  --data_dir datasets \
  --dataset bbbp \
  --delocalization \
  --runs 10 \
  --seed 42 \
  --device 0 \
  --epochs 300 \
  --batch_size 32 \
  --lr 0.000332 \
  --min_lr 0.0001 \
  --wd 0.000209 \
  --log_steps 1 \
  --method mhnn \
  --All_num_layers 3 \
  --MLP1_num_layers 3 \
  --MLP2_num_layers 2 \
  --MLP3_num_layers 2 \
  --MLP4_num_layers 2 \
  --MLP_hidden 64 \
  --output_num_layers 2 \
  --output_hidden 173 \
  --aggregate mean \
  --normalization ln \
  --activation prelu \
  --dropout 0.001666