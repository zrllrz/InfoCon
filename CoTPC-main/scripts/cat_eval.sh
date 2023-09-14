#!/bin/bash

TASK="TurnFaucet-v0"
MODEL_NAME="TF-0912-LONG"

cd ../src/$TASK"_eval/" &&
cat $MODEL_NAME"/eval_output.txt"
echo "################"
ls
echo "################"