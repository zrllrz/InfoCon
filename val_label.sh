#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=7 python val_label.py \
   --task=PegInsertionSide-v0 \
   --control_mode=pd_joint_delta_pos \
   --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=6_29_1-l_30-b_0.2-ks_a-kn_3-an_3-rn_3-prevention \
   --from_ckpt=2000 \
