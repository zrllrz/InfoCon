#!/bin/bash

TASK=PickCube-v0
MODEL_NAME=PC_TEST
I=10

cd ../src &&

# Example script for PickCube CoTPC evaluation
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=$I \
  --task=$TASK \
  --model_name=$MODEL_NAME \
  --n_env=25
