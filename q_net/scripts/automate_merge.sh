#!/bin/bash

BASE_PATH="embeddings"
RESULTS_DIR="results"  # Directory where results will be saved

mkdir -p $RESULTS_DIR


for folder in $BASE_PATH/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        
        echo "Running SVC model on $folder_name"
        python holdout_validation.py --data_path "$folder_name" --model svc
        
        echo "Running MLP model on $folder_name"
        python holdout_validation.py --data_path "$folder_name" --model mlp --total_num_seed 20
        
        model_results_dir="$RESULTS_DIR/$folder_name"
        mkdir -p $model_results_dir

        for model in svc mlp; do
            for seed in $(seq 0 19); do
                src_file=$(find "$folder_name" -path "*$model/$folder_name/seed_${seed}/average_metrics_${model}.csv")
                if [ -f "$src_file" ]; then
                    cp "$src_file" "$model_results_dir/"
                fi
            done
        done
    fi
done

# Merge results
echo "Merging results"
python merge_results.py --results_dir $RESULTS_DIR