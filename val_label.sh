#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=7 python val_label.py \
   --task=PegInsertionSide-v0 \
   --control_mode=pd_joint_delta_pos \
   --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=6_27_0-l_30-b_0.2-ks_a-prevention \
   --from_ckpt=6000 \
