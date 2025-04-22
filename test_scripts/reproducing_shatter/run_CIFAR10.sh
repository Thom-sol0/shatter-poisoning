Copy

#!/bin/bash

set -euxo pipefail

# Initialize conda for the shell
source ${CONDA_PREFIX}/etc/profile.d/conda.sh

# Now activate your environment
conda activate shatter
export SHATTER_HOME="$(pwd)"

echo "Computing EL on CIFAR10"
cd $SHATTER_HOME/test_scripts/reproducing_shatter/CIFAR10
$SHATTER_HOME/eval/run_helper.sh 8 300 $(pwd)/config_EL.ini $SHATTER_HOME/eval/testingSimulation.py 10 10 $SHATTER_HOME/eval/data/CIFAR10 $SHATTER_HOME/eval/data/CIFAR10

echo "Computing Muffliato on CIFAR10"
$SHATTER_HOME/eval/run_helper.sh 8 300 $(pwd)/config_Muffliato05.ini $SHATTER_HOME/eval/testingMuffliato.py 10 10 $SHATTER_HOME/eval/data/CIFAR10 $SHATTER_HOME/eval/data/CIFAR10

echo "Computing Shatter on CIFAR10"
$SHATTER_HOME/eval/run_helper.sh 8 300 $(pwd)/config_Shatter2.ini $SHATTER_HOME/eval/testingSimulation.py 10 10 $SHATTER_HOME/eval/data/CIFAR10 $SHATTER_HOME/eval/data/CIFAR10

echo "Making CSVs and Plots for CIFAR10 in ./CIFAR10"
python $SHATTER_HOME/eval/plot_csv_acc.py .
python $SHATTER_HOME/eval/evaluate_attack.py .

echo "Done CIFAR10!"
