#!/usr/bin/env python3

import os
import argparse
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_l2_norm_diff(model_path1, model_path2):
    """
    Compute L2 norm difference between two saved model state dictionaries
    """
    state1 = torch.load(model_path1)
    state2 = torch.load(model_path2)
    
    total_squared_diff = 0.0
    layer_norms = {}
    
    for key in state1:
        if key in state2:
            diff = state1[key] - state2[key]
            layer_norm = torch.norm(diff).item()
            layer_norms[key] = layer_norm
            total_squared_diff += layer_norm ** 2
    
    total_norm = np.sqrt(total_squared_diff)
    return total_norm, layer_norms

def find_experiment_dirs(base_dir):
    """
    Find all experiment directories in the base directory
    """
    exp_dirs = []
    for path in Path(base_dir).glob("exp_*"):
        if path.is_dir():
            exp_dirs.append(path)
    
    return sorted(exp_dirs)

def compare_experiments(exp_dir1, exp_dir2, node_id=None):
    """
    Compare two experiment directories and generate analysis
    """
    # Find all nodes if node_id is not specified
    if node_id is None:
        nodes = set()
        for exp_dir in [exp_dir1, exp_dir2]:
            for path in Path(exp_dir).glob("node*_round*.pt"):
                node = int(path.name.split("node")[1].split("_")[0])
                nodes.add(node)
        nodes = sorted(nodes)
    else:
        nodes = [node_id]
    
    results = []
    
    # Compare each round for each node
    for node in nodes:
        node_results = []
        files1 = sorted(Path(exp_dir1).glob(f"node{node}_round*.pt"))
        rounds = [int(f.name.split("round")[1].split(".pt")[0]) for f in files1]
        
        for round_num in rounds:
            file1 = os.path.join(exp_dir1, f"node{node}_round{round_num}.pt")
            file2 = os.path.join(exp_dir2, f"node{node}_round{round_num}.pt")
            
            if os.path.exists(file1) and os.path.exists(file2):
                total_norm, _ = compute_l2_norm_diff(file1, file2)
                node_results.append({
                    "node": node,
                    "round": round_num,
                    "l2_norm": total_norm
                })
        
        if node_results:
            results.extend(node_results)
    
    return pd.DataFrame(results)

def plot_l2_norm_comparison(df, output_path=None, show_plot=True):
    """
    Plot L2 norm differences over rounds
    """
    plt.figure(figsize=(12, 6))
    
    for node in df["node"].unique():
        node_data = df[df["node"] == node]
        plt.plot(node_data["round"], node_data["l2_norm"], label=f"Node {node}")
    
    plt.xlabel("Communication Round")
    plt.ylabel("L2 Norm Difference")
    plt.title("Model Weight Difference Between Experiments")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Compare model weights between different experiment runs"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./saved_weights",
        help="Base directory containing experiment folders"
    )
    
    parser.add_argument(
        "--exp1",
        type=str,
        required=True,
        help="First experiment ID to compare"
    )
    
    parser.add_argument(
        "--exp2",
        type=str,
        required=True,
        help="Second experiment ID to compare"
    )
    
    parser.add_argument(
        "--node",
        type=int,
        default=None,
        help="Specific node ID to analyze (default: analyze all nodes)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="weight_diff_comparison.pdf",
        help="Output file path for the plot"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="weight_diff_comparison.csv",
        help="Output file path for the CSV results"
    )
    
    args = parser.parse_args()
    
    # Construct experiment directory paths
    exp_dir1 = os.path.join(args.base_dir, args.exp1)
    exp_dir2 = os.path.join(args.base_dir, args.exp2)
    
    # Verify directories exist
    if not os.path.exists(exp_dir1):
        print(f"Error: Experiment directory not found: {exp_dir1}")
        return 1
    
    if not os.path.exists(exp_dir2):
        print(f"Error: Experiment directory not found: {exp_dir2}")
        return 1
    
    # Compare experiments
    print(f"Comparing experiments:\n  1. {args.exp1}\n  2. {args.exp2}")
    df = compare_experiments(exp_dir1, exp_dir2, args.node)
    
    if df.empty:
        print("Error: No matching model weights found for comparison")
        return 1
    
    # Save results to CSV
    df.to_csv(args.csv, index=False)
    print(f"Results saved to CSV: {args.csv}")
    
    # Plot results
    plot_l2_norm_comparison(df, args.output)
    print(f"Plot saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
