#!/bin/bash

# Define the root path of the Conda environment
CONDA_ENV_PATH=$(dirname $(which python))/..

# Find all Deepspeed paths under different Python versions and apply changes
find $CONDA_ENV_PATH/lib/python* -type d -name "deepspeed" | while read deepspeed_path; do
  utils_file="$deepspeed_path/runtime/utils.py"
  stage_file="$deepspeed_path/runtime/zero/stage_1_and_2.py"
  
  # Check if files exist and then perform sed operations
  if [ -f "$utils_file" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' "$utils_file"
    echo 'replaced from torch._six import inf by from torch import inf in' $utils_file
  fi
  if [ -f "$stage_file" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' "$stage_file"
    echo 'replaced from torch._six import inf by from torch import inf in' $utils_file
  fi
done
