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

# Debugging output
echo "Source directory: ${source_dir}"
echo "Target directory: ${target_dir}"
echo "Keyword: ${keyword}"

# Create target directory if it doesn't exist
mkdir -p "${target_dir}"

# Function to process a directory
process_directory() {
    local dir="$1"
    local dest_dir="$2"

    # Flag to check if files are found
    local files_found=false

    # Loop through each item in the directory
    for item in "${dir}"/*; do
        if [ -d "${item}" ]; then
            # If it's a directory, recurse into it
            process_directory "${item}" "${dest_dir}/$(basename "${item}")"
        elif [ -f "${item}" ]; then
            # If it's a file, process it
            files_found=true
            image_name=$(basename "${item}")
            new_image_name="${image_name%.*}_${keyword}.${image_name##*.}"
            echo "Processing file: ${item}"
            echo "Renaming to: ${new_image_name}"
            # Create the target directory if it doesn't exist
            mkdir -p "$(dirname "${dest_dir}/${new_image_name}")"
            cp "${item}" "${dest_dir}/${new_image_name}"
        fi
    done

    # Inform if no files were found in the directory
    if [ "$files_found" = false ]; then
        echo "No files found in ${dir}"
    fi
}

# Loop through each class directory
for class_dir in "${source_dir}"/*; do
    if [ -d "${class_dir}" ]; then
        class=$(basename "${class_dir}")
        echo "Processing class directory: ${class_dir}"
        
        # Create corresponding class directory in target directory
        mkdir -p "${target_dir}/${class}"
        
        # Process the class directory recursively
        process_directory "${class_dir}" "${target_dir}/${class}"
    else
        echo "No class directories found in ${source_dir}"
    fi
done

echo "Renaming and copying of images completed."