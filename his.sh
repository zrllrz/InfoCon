#!/bin/bash

cd src &&

python his.py \
  --task=PegInsertionSide-v0 \
  --control_mode=pd_joint_delta_pos \
  --obs_mode=state \
  --seed=0 \
  --key_name=keys.txt