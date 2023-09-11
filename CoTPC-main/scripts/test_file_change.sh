#!/bin/bash

cd ../src &&

FILE_DIR="../save_model/"
MODEL_NAME="SC-0903/"
i=100

echo "$FILE_DIR$MODEL_NAME$i.pth"
if test -e "$FILE_DIR$MODEL_NAME$i.pth"; then
  echo "exist"
else
  echo "un-exist"
fi
