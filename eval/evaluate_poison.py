#!/usr/bin/env python3

import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_results(results_dir):
    """Load results from CSV files in directory"""
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    dfs = []
    
    for csv_file in csv_files:
        try:
            # Extract node ID from filename
            node_id = int(os.path.basename(csv_file).split('_')[0])
            df = pd.read_csv(csv_file)
            df['node_id'] = node_id
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if dfs:
        return pd.concat(dfs)
    return None

def load_poison_metrics(results_dir):
    """Load poisoning metrics from JSON files in directory"""
    json_files = glob.glob(os.path.join(results_dir, "poison_metrics_*.json"))
    metrics = []
    
    for json_file in json_files:
        try:
            # Extract node ID from filename
            node_id = int(os.path.basename(json_file).split('_')[2].split('.')[0])
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['node_id'] = node_id
                metrics.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return metrics

def plot_accuracy_comparison(baseline_df, poisoned_df, attack_type, output_dir):
    """Plot accuracy comparison between baseline and poisoned runs"""
    plt.figure(figsize=(10, 6))
    
    # Group by iteration and compute mean/std
    baseline_grouped = baseline_df.groupby('iteration')['test_acc'].agg(['mean', 'std']).reset_index()
    poisoned_grouped = poisoned_df.groupby('iteration')['test_acc'].agg(['mean', 'std']).reset_index()
    
    # Plot with error bars
    plt.errorbar(baseline_grouped['iteration'], baseline_grouped['mean'], 
                yerr=baseline_grouped['std'], label='Baseline', marker='o', capsize=4)
    plt.errorbar(poisoned_grouped['iteration'], poisoned_grouped['mean'], 
                yerr=poisoned_grouped['std'], label=f'{attack_type} Attack', marker='s', capsize=4)
    
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title(f'Impact of {attack_type} Poisoning Attack on Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make x-axis integers only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{attack_type}.png'))
    plt.close()

def plot_loss_comparison(baseline_df, poisoned_df, attack_type, output_dir):
    """Plot loss comparison between baseline and poisoned runs"""
    plt.figure(figsize=(10, 6))
    
    # Group by iteration and compute mean/std
    baseline_grouped = baseline_df.groupby('iteration')['test_loss'].agg(['mean', 'std']).reset_index()
    poisoned_grouped = poisoned_df.groupby('iteration')['test_loss'].agg(['mean', 'std']).reset_index()
    
    # Plot with error bars
    plt.errorbar(baseline_grouped['iteration'], baseline_grouped['mean'], 
                yerr=baseline_grouped['std'], label='Baseline', marker='o', capsize=4)
    plt.errorbar(poisoned_grouped['iteration'], poisoned_grouped['mean'], 
                yerr=poisoned_grouped['std'], label=f'{attack_type} Attack', marker='s', capsize=4)
    
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss')
    plt.title(f'Impact of {attack_type} Poisoning Attack on Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make x-axis integers only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_comparison_{attack_type}.png'))
    plt.close()

def plot_poison_metrics(metrics, attack_type, output_dir):
    """Plot poisoning metrics"""
    if not metrics:
        print("No poisoning metrics found")
        return
    
    poison_rates = [m.get('poison_rate', 0) for m in metrics]
    node_ids = [m.get('node_id', 0) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar(node_ids, poison_rates)
    plt.xlabel('Node ID')
    plt.ylabel('Poison Rate')
    plt.title(f'Poisoning Rate by Node ({attack_type} Attack)')
    plt.xticks(node_ids)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'poison_rate_{attack_type}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate poisoning attack results')
    parser.add_argument('--baseline-dir', required=True, help='Directory with baseline results')
    parser.add_argument('--poisoned-dir', required=True, help='Directory with poisoned results')
    parser.add_argument('--attack-type', default='zero', help='Type of poisoning attack')
    parser.add_argument('--output-dir', default='./attack_evaluation', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    baseline_df = load_results(args.baseline_dir)
    poisoned_df = load_results(args.poisoned_dir)
    poison_metrics = load_poison_metrics(args.poisoned_dir)
    
    if baseline_df is None or poisoned_df is None:
        print("Error: Could not load results files")
        return
        
    # Generate plots
    plot_accuracy_comparison(baseline_df, poisoned_df, args.attack_type, args.output_dir)
    plot_loss_comparison(baseline_df, poisoned_df, args.attack_type, args.output_dir)
    plot_poison_metrics(poison_metrics, args.attack_type, args.output_dir)
    
    # Print summary statistics
    print(f"\n=== Attack Impact Summary ({args.attack_type}) ===")
    
    # Get final accuracy
    final_baseline_acc = baseline_df[baseline_df['iteration'] == baseline_df['iteration'].max()]['test_acc'].mean()
    final_poisoned_acc = poisoned_df[poisoned_df['iteration'] == poisoned_df['iteration'].max()]['test_acc'].mean()
    
    print(f"Final baseline accuracy: {final_baseline_acc:.4f}")
    print(f"Final poisoned accuracy: {final_poisoned_acc:.4f}")
    print(f"Accuracy impact: {final_poisoned_acc - final_baseline_acc:.4f}")
    print(f"Relative accuracy change: {(final_poisoned_acc - final_baseline_acc) / final_baseline_acc * 100:.2f}%")
    
    # Calculate convergence metrics if available
    if 'total_round_time_no_eval' in baseline_df.columns and 'total_round_time_no_eval' in poisoned_df.columns:
        baseline_time = baseline_df['total_round_time_no_eval'].sum()
        poisoned_time = poisoned_df['total_round_time_no_eval'].sum()
        print(f"Time impact: {poisoned_time - baseline_time:.2f}s")

if __name__ == '__main__':
    main()
