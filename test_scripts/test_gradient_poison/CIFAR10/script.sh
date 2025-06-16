#!/bin/bash
# script.sh - Run sequentially for all .ini config files

# Get absolute path to current directory (test_scripts/box_plot/CIFAR10)
CONFIG_DIR="$PWD"

# Loop through all .ini files in the current directory
for config_file in "$CONFIG_DIR"/*.ini; do
    echo "========================================================================"
    echo "Processing config file: $(basename "$config_file")"
    echo "========================================================================"
    
    # Run the command with the current config file
    "$SHATTER_HOME/eval/run_helper.sh" 16 301 "$config_file" \
        "$SHATTER_HOME/eval/testingSimulation.py" 10 10 \
        "$SHATTER_HOME/eval/data/CIFAR10" \
        "$SHATTER_HOME/eval/data/CIFAR10"
    
    echo "Finished processing $(basename "$config_file")"
    echo
done

echo "All configurations completed!"