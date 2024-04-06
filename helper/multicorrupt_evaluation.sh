#!/bin/bash

# List of corruptions and severity levels
corruptions=("beamsreducing" "brightness" "dark" "fog" "missingcamera" "motionblur" "pointsreducing" "snow" "spatialmisalignment" "temporalmisalignment")
severity_levels=("1" "2" "3")

# Directory paths
multicorrupt_root="/workspace/multicorrupt/"
nuscenes_data_dir="/workspace/data/nuscenes"
logfile="/workspace/evaluation_log.txt"

# Model evaluation command (replace with your actual command)

# Loop over corruptions and severity levels
for corruption in "${corruptions[@]}"; do
  for severity in "${severity_levels[@]}"; do
    # Log the current configuration in the terminal
    echo "Current Configuration: Corruption=$corruption, Severity=$severity"

    # Create soft link in /workspace/data/nuscenes
    ln -s "$multicorrupt_root/$corruption/$severity" "$nuscenes_data_dir"

    # Perform model evaluation
    output=$(bash tools/test.py /workspace/projects/configs/path_to_config.py /workspace/ckpts/path_to_checkpoint.pth)

   # Save the entire output to a separate text file
    echo "$output" > "/workspace/${corruption}_${severity}_output.txt"

    # Extract NDS and mAP scores from the output
    nds=$(echo "$output" | grep "NDS:" | awk '{print $2}')
    map=$(echo "$output" | grep "mAP:" | awk '{print $2}')

    # Save results to the logfile
    echo "Corruption: $corruption, Severity: $severity, NDS: $nds, mAP: $map" >> "$logfile"

    # Remove soft link
    rm "$nuscenes_data_dir"
  done
done
