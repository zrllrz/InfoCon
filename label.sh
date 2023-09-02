#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=5 python label.py \
   --task=StackCube-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=-1 \
   --model_name=SC_0901_1630_no-rec-temb_GPTHN_NN_EMA_DGOAL_GGOAL_RECSHARD_ENCO+_INCNLLk4-r4-f2-c10_KT0.1_EMA0.9_ema_ave_st-emb1.2-r_l10.0-egpthn_s2_a1-emb128-key128-e128-cluster0.001-rec0.1 \
   --from_ckpt=2000 \



