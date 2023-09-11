#!/bin/bash

FILE_DIR="model_checkpoints/"
MODEL_NAME="TF_0908_0755_LONG_k4-r4-f2-c10_KT0.1_EMA0.9_ema_ave_st-emb1.2-r_l10.0-use_r-egpthn_s2_a1-emb128-key128-e128-cluster0.001-rec0.1-finetune/"
I=100

cd src &&

while [[ $I -le 16000 ]];
do
  if test -e "$FILE_DIR$MODEL_NAME""/epoch""$I.pth"; then
    echo "$I""exist"
    ((I+=100))
  fi
done
