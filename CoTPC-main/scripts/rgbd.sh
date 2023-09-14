#!/bin/bash

TASK="StackCube-v0"

cd ../src &&

for ((i=0; i<100; i++));do
  CUDA_VISIBLE_DEVICES=0 python rgbd.py \
    --task=$TASK \
    --idx=$i \
    --use_hand_camera
done