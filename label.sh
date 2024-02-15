#!/bin/bash

# setting your own task name, model name, and checkpoint
TASK=PickCube-v0
MODEL_NAME=PC_TEST
I=10
KEY_NAME=key_test.txt

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
  --task=$TASK \
  --control_mode=pd_joint_delta_pos \
  --obs_mode=state \
  --seed=0 \
  --n_traj=-1 \
  --model_name=$MODEL_NAME \
  --from_ckpt=$I \
  --key_name=$KEY_NAME
