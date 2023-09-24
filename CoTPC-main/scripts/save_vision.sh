#!/bin/bash

cd ../src &&

#CUDA_VISIBLE_DEVICES=2 python save_vision.py \
#  --task=$TASK

CUDA_VISIBLE_DEVICES=2 python save_vision.py \
  --task=PickCube-v0