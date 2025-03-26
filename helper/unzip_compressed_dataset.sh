#!/bin/bash

# Set directories
compressed_dir="multicorrupt"
destination_dir="multicorrupt_uncompressed"
mkdir -p "$destination_dir"

# Use find to locate all .part00 files recursively
# Use print0 and read -d $'\0' for robust handling of filenames with spaces/special chars
find "$compressed_dir" -type f -name '*.tar.gz.part00' -print0 | while IFS= read -r -d $'\0' archive; do
    
    # Get the directory containing the archive parts
    part_dir=$(dirname "$archive")
    
    # Get the base name without the suffix and path (e.g., beamsreducing_1)
    base_name=$(basename "$archive" .tar.gz.part00)
    
    # Extract category and subfolder from the base name
    # Assuming format category_subfolder
    category=$(echo "$base_name" | cut -d'_' -f1)
    subfolder=$(echo "$base_name" | cut -d'_' -f2)

    # Check if category and subfolder were extracted correctly
    if [[ -z "$category" || -z "$subfolder" || "$category" == "$subfolder" ]]; then
        echo "Warning: Could not parse category/subfolder from '$base_name' found in '$archive'. Skipping."
        continue
    fi
    
    # Define the target directory for extraction
    target_subdir="$destination_dir/$category/$subfolder"
    
    # Create target category/subfolder directory if it doesn't exist
    mkdir -p "$target_subdir"
    
    echo "Reconstructing and extracting $base_name (from $part_dir) into $target_subdir ..."
    
    # Concatenate all parts matching the base_name within their specific directory
    # and pipe to tar for extraction into the correct destination subfolder
    cat "$part_dir/${base_name}.tar.gz.part"* | tar -xzvf - -C "$target_subdir"

    # Optional: Check tar's exit status
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to extract '$base_name'. Check archive parts in '$part_dir'."
        # Decide if you want to exit or continue:
        # exit 1 
    fi

done

echo "Dataset extraction completed."