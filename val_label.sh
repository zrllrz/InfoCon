#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=7 python val_label.py \
   --task=PegInsertionSide-v0 \
   --control_mode=pd_joint_delta_pos \
   --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=skip-l_20-b_0.2-ks_a \
   --from_ckpt=10000 \
