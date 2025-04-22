#!/bin/bash

set -euxo pipefail

# Initialize conda for the shell
if [ -n "${CONDA_PREFIX+x}" ]; then
    source ${CONDA_PREFIX}/etc/profile.d/conda.sh
    # Activate environment
    conda activate shatter
fi

# Use the same SHATTER_HOME definition as in the existing scripts
export SHATTER_HOME="$(pwd)/../../../"

echo "Setting up data directories"
mkdir -p ./data/CIFAR10
ln -sf $SHATTER_HOME/eval/data/CIFAR10/* ./data/CIFAR10/ 2>/dev/null || true

# Create configs directory if it doesn't exist
mkdir -p ./configs

# Run baseline experiment (first test with small number of iterations)
echo "Running baseline (non-poisoned) experiment on CIFAR10"
$SHATTER_HOME/eval/run_helper.sh 4 2 $(pwd)/config_cifar_poison.ini $SHATTER_HOME/eval/testingSimulation.py 1 1 $SHATTER_HOME/eval/data/CIFAR10 $SHATTER_HOME/eval/data/CIFAR10

# If the small test succeeded, we can generate the poisoning configs
echo "Creating poisoning configurations"

# Zero gradients attack
echo "Generating zero gradient attack config"
cp $(pwd)/config_cifar_poison.ini $(pwd)/configs/config_poison_zero.ini
sed -i 's/attack_type = zero/attack_type = zero/' $(pwd)/configs/config_poison_zero.ini

# Gradient flipping attack
echo "Generating gradient flipping attack config"
cp $(pwd)/config_cifar_poison.ini $(pwd)/configs/config_poison_flip.ini
sed -i 's/attack_type = zero/attack_type = flip/' $(pwd)/configs/config_poison_flip.ini

# Noise injection attack
echo "Generating noise injection attack config"
cp $(pwd)/config_cifar_poison.ini $(pwd)/configs/config_poison_noise.ini
sed -i 's/attack_type = zero/attack_type = noise/' $(pwd)/configs/config_poison_noise.ini
sed -i 's/poison_strength = 1.0/poison_strength = 5.0/' $(pwd)/configs/config_poison_noise.ini

echo "Generated poisoning configurations successfully!"
echo "To run experiments, use:"
echo "$SHATTER_HOME/eval/run_helper.sh 8 300 \$(pwd)/configs/config_poison_zero.ini \$SHATTER_HOME/eval/testingSimulation.py 10 10 \$SHATTER_HOME/eval/data/CIFAR10 \$SHATTER_HOME/eval/data/CIFAR10"
echo "$SHATTER_HOME/eval/run_helper.sh 8 300 \$(pwd)/configs/config_poison_flip.ini \$SHATTER_HOME/eval/testingSimulation.py 10 10 \$SHATTER_HOME/eval/data/CIFAR10 \$SHATTER_HOME/eval/data/CIFAR10"
echo "$SHATTER_HOME/eval/run_helper.sh 8 300 \$(pwd)/configs/config_poison_noise.ini \$SHATTER_HOME/eval/testingSimulation.py 10 10 \$SHATTER_HOME/eval/data/CIFAR10 \$SHATTER_HOME/eval/data/CIFAR10"