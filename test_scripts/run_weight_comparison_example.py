#!/usr/bin/env python3
"""
Example of how to use the VNodeSharingDefenseNoneWithWeightTracking class
for comparing model weights between runs with different numbers of attackers.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from virtualNodes.mappings.VNodeLinear import VNodeLinear
from virtualNodes.sharing.VNodeSharingDefenseNoneWithWeightTracking import VNodeSharingDefenseNoneWithWeightTracking

def run_experiment(config_path, attackers_count, save_interval=10):
    """
    Run an experiment with the specified number of attackers
    """
    # Read configuration
    config = LocalConfig(config_path)
    
    # Modify the config to set the number of attackers
    config["sharing"]["params"]["adversarial_nodes"] = ",".join([str(i) for i in range(attackers_count)])
    
    # Set experiment ID based on attacker count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}_atk{attackers_count}"
    
    # Modify the config to use weight tracking
    config["sharing"]["class"] = "virtualNodes.sharing.VNodeSharingDefenseNoneWithWeightTracking.VNodeSharingDefenseNoneWithWeightTracking"
    
    # Add save_interval and experiment_id to config
    config["sharing"]["params"]["save_interval"] = save_interval
    config["sharing"]["params"]["experiment_id"] = experiment_id
    
    # Save the modified config to a temporary file
    temp_config_path = f"temp_config_{attackers_count}.ini"
    with open(temp_config_path, "w") as f:
        config.write(f)
    
    # TODO: Add code to run your experiment with the modified config
    # This will depend on your specific setup for running experiments
    
    return experiment_id, temp_config_path

def compare_experiments(exp_id1, exp_id2, base_dir="./saved_weights"):
    """
    Compare two experiments using the comparison tool
    """
    script_path = os.path.join(os.path.dirname(__file__), "compare_model_weights.py")
    
    cmd = f"{script_path} --base-dir {base_dir} --exp1 {exp_id1} --exp2 {exp_id2}"
    
    print(f"Running comparison command: {cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different numbers of attackers and compare L2 norms"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base configuration file"
    )
    
    parser.add_argument(
        "--attackers1",
        type=int,
        default=1,
        help="Number of attackers for the first experiment"
    )
    
    parser.add_argument(
        "--attackers2",
        type=int,
        default=5,
        help="Number of attackers for the second experiment"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval (in rounds) at which to save model weights"
    )
    
    args = parser.parse_args()
    
    # Run first experiment
    print(f"Running experiment with {args.attackers1} attackers...")
    exp_id1, config1 = run_experiment(args.config, args.attackers1, args.interval)
    
    # Run second experiment
    print(f"Running experiment with {args.attackers2} attackers...")
    exp_id2, config2 = run_experiment(args.config, args.attackers2, args.interval)
    
    # Compare experiments
    print("Comparing experiment results...")
    compare_experiments(exp_id1, exp_id2)
    
    # Clean up temporary config files
    os.remove(config1)
    os.remove(config2)
    
    print("Done!")
    
if __name__ == "__main__":
    main()
