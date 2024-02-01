#!/bin/bash

# setting your own task name, model name, and checkpoint
TASK=
MODEL_NAME=
I=

cd src &&

CUDA_VISIBLE_DEVICES=2 python label.py \
  --task=$TASK --control_mode=pd_joint_delta_pos --obs_mode=state \
  --seed=0 \
  --n_traj=-1 \
  --model_name=$MODEL_NAME \
  --from_ckpt=$I \
