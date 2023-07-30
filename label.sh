#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=rkac-elastic-zerots-contrast-legacy-code_10-subgoal-detach-persistence_none_wutreset-keg_0.1 \
   --from_ckpt=1000 \
