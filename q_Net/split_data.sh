#!/bin/bash

# Usage message for help
usage() {
    echo "Usage: $0 -s <source_dir> -t <target_dir> -r <split_ratio>"
    echo "  -s  Source directory containing the class folders"
    echo "  -t  Target directory where the split data will be stored"
    echo "  -r  Split ratio for training data (e.g., 80 for 80% training, 20% validation)"
    exit 1
}

# Default values
split_ratio=80

# Parse command-line arguments
while getopts "s:t:r:" opt; do
    case "${opt}" in
        s)
            source_dir="${OPTARG}"
            ;;
        t)
            target_dir="${OPTARG}"
            ;;
        r)
            split_ratio="${OPTARG}"
            ;;
        *)
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "${source_dir}" ] || [ -z "${target_dir}" ] || [ -z "${split_ratio}" ]; then
    usage
fi

# Define target directories
kcross_dir="${target_dir}/train"
test_dir="${target_dir}/val"

# Create target directories
mkdir -p "${kcross_dir}"
mkdir -p "${test_dir}"

# Loop through each class directory
for class_dir in "${source_dir}"/*; do
    if [ -d "${class_dir}" ]; then
        class=$(basename "${class_dir}")
        
        # Create class directories in target directories
        mkdir -p "${kcross_dir}/${class}"
        mkdir -p "${test_dir}/${class}"

        # Get list of all images in class directory
        images=("${class_dir}"/*.png)
        total_images=${#images[@]}
        kcross_count=$((total_images * split_ratio / 100))

        # Shuffle the images array
        for ((i = 0; i < total_images; i++)); do
            j=$((RANDOM % (i + 1)))
            tmp=${images[i]}
            images[i]=${images[j]}
            images[j]=$tmp
        done

        # Copy split_ratio% of images to kcross directory
        for ((i = 0; i < kcross_count; i++)); do
            cp "${images[$i]}" "${kcross_dir}/${class}/"
        done

        # Copy remaining images to test directory
        for ((i = kcross_count; i < total_images; i++)); do
            cp "${images[$i]}" "${test_dir}/${class}/"
        done
    fi
done

echo "Data split into train and validation directories completed."