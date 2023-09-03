#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=6 python label.py \
   --task=TurnFaucet-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=-1 \
   --model_name=TF_0902_1002k4-r4-f2-c10_KT0.1_EMA0.9_ema_ave_st-emb1.2-r_l10.0-use_r-egpthn_s2_a1-emb128-key128-e128-cluster0.001-rec0.1-finetune \
   --from_ckpt=4000 \
   --pause


# 0902
#       PegInsertionSide    StackCube   TurnFaucet
# 2000
# 2100
# 2200
# 2300
# 2400
# 2500
# 2600
# 2700  23.47
# 2800  33.35
# 2900  37.96
# 3000  25.67                           \
# 3100  22.37                           35.342288557213934
# 3200  22.33                           38.208955223880594
# 3300  25.52                           33.15422885572139
# 3400  26.57                           35.6089552238806
# 3500  33.61                           35.744278606965175
# 3600  36.89                           48.12238805970149
# 3700  32.16                           34.73
# 3800  32.73                           28.60
# 3900  27.53               09.30       28.60
# 4000  32.10               16.93       23.81

# 0903
#       PegInsertionSide    StackCube   TurnFaucet
# 3000
# 3100
# 3200
# 3300
# 3400
# 3500
# 3600
# 3700
# 3800
# 3900
# 4000
