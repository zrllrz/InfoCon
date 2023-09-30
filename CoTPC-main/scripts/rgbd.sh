#!/bin/bash

TASK=TurnFaucet-v0

cd ../src &&

for i in 0 4 5 102 106 201 209 503 504 307 317 613 617 801 809; do
  CUDA_VISIBLE_DEVICES=6 python rgbd.py \
    --task=$TASK \
    --idx=$i \
    --use_hand_camera
done

#for i in 20 26 50 51 52 60 64 68 82 89 99; do
#  CUDA_VISIBLE_DEVICES=5 python rgbd.py \
#    --task=$TASK \
#    --idx=$i \
#    --use_hand_camera
#done

#for i in 20 22 33 42 62 75 83 85 86 93 96 97; do
#  CUDA_VISIBLE_DEVICES=5 python rgbd.py \
#    --task=$TASK \
#    --idx=$i \
#    --use_hand_camera
#done