#!/bin/bash

TASK="StackCube-v0"

cd ../src &&

CUDA_VISIBLE_DEVICES=4 python rgbd.py \
      --task=$TASK \
      --idx=0