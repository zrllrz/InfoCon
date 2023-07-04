#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=7 python val_label2.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=independent_7_3_0 \
   --from_ckpt=3000 \
