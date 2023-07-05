#!/bin/bash

cd src &&


CUDA_VISIBLE_DEVICES=7 python val_label2.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
   --from_ckpt=5000 \
