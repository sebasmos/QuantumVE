#!/bin/bash

# Check if the base path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <BASE_PATH>"
    exit 1
fi

BASE_PATH="$1"

for folder in "$BASE_PATH"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        
        python ../part2_Classic_classification/holdout_validation.py --data_path "$folder" --model mlp --total_num_seed 20
        python ../part2_Classic_classification/holdout_validation.py --data_path "$folder" --model svc --total_num_seed 20
        
    fi
done