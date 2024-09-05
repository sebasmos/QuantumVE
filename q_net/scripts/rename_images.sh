#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 -s <source_dir> -t <target_dir> -k <keyword>"
    echo "  -s  Source directory containing the class folders"
    echo "  -t  Target directory where the renamed images will be saved"
    echo "  -k  Keyword to append to the image filenames"
    exit 1
}

# Parse command-line arguments
while getopts "s:t:k:" opt; do
    case "${opt}" in
        s)
            source_dir="${OPTARG}"
            ;;
        t)
            target_dir="${OPTARG}"
            ;;
        k)
            keyword="${OPTARG}"
            ;;
        *)
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "${source_dir}" ] || [ -z "${target_dir}" ] || [ -z "${keyword}" ]; then
    usage
fi

# Create target directory if it doesn't exist
mkdir -p "${target_dir}"

# Loop through each class directory
for class_dir in "${source_dir}"/*; do
    if [ -d "${class_dir}" ]; then
        class=$(basename "${class_dir}")
        
        # Create corresponding class directory in target directory
        mkdir -p "${target_dir}/${class}"
        
        # Loop through each image in the class directory
        for image_file in "${class_dir}"/*; do
            if [ -f "${image_file}" ]; then
                # Get the base name of the image file
                image_name=$(basename "${image_file}")
                # Create the new image name with the keyword appended
                new_image_name="${image_name%.*}_${keyword}.${image_name##*.}"
                # Copy and rename the image to the target directory
                cp "${image_file}" "${target_dir}/${class}/${new_image_name}"
            fi
        done
    fi
done

echo "Renaming and copying of images completed."