#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=3 python val_pc.py \
   --task=PegInsertionSide-v0 \
   --control_mode=pd_joint_delta_pos \
   --obs_mode=state \
   --seed=0 \
   --n_traj=100 \
   --model_name=l_10-al_2 \
   --from_ckpt=8000 \
