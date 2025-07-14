import os
import sys
from pathlib import Path
import configparser
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MAX_ITERATION = None
MUFFLIATO_ROUNDS = 10


def extract_adversarial_nodes(config_path):
    """Extract adversarial nodes from config file"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        return []
        
    config = configparser.ConfigParser()
    config.read(config_path)
    
    try:
        if 'SHARING' in config and 'adversarial_nodes' in config['SHARING']:
            nodes_str = config['SHARING']['adversarial_nodes']
            return [int(node) for node in nodes_str.split(',')]
    except Exception as e:
        print(f"Error extracting adversarial nodes from config: {e}")
    
    return []


def get_stats(l, adversarial_nodes=None, node_ids=None):
    """Calculate statistics excluding adversarial nodes and properly handling NaN values"""
    assert len(l) > 0
    mean_dict, stdev_dict, min_dict, max_dict, counts_dict = {}, {}, {}, {}, {}
    
    # Find all unique keys across all dataframes
    all_keys = set()
    for df in l:
        all_keys.update(df.index.tolist())
    
    # Filter out outlier keys
    filtered_keys = filter_outlier_keys(all_keys)
    
    for key in sorted(filtered_keys):  # Process keys in order
        if MAX_ITERATION is not None and key >= MAX_ITERATION:
            continue
            
        # Filter to include only defender nodes if node_ids are provided
        if node_ids and adversarial_nodes:
            # Only include values for dataframes that have this key
            all_nodes = []
            for i, node_id in zip(l, node_ids):
                if node_id not in adversarial_nodes and key in i.index:
                    all_nodes.append(i[key])
        else:
            # Only include values for dataframes that have this key
            all_nodes = [i[key] for i in l if key in i.index]
            
        if not all_nodes:  # Skip if no valid values found
            continue
            
        all_nodes = np.array(all_nodes)
        non_nan_count = np.count_nonzero(~np.isnan(all_nodes))
        
        # Even if values are all NaN, still record the statistics
        mean = np.nanmean(all_nodes) if non_nan_count > 0 else float('nan')
        std = np.nanstd(all_nodes) if non_nan_count > 0 else float('nan')
        min_val = np.nanmin(all_nodes) if non_nan_count > 0 else float('nan')
        max_val = np.nanmax(all_nodes) if non_nan_count > 0 else float('nan')
        
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min_val
        max_dict[int(key)] = max_val
        counts_dict[int(key)] = non_nan_count
        
    return mean_dict, stdev_dict, min_dict, max_dict, counts_dict

def filter_outlier_keys(keys):
    """Filter out outlier indices that could mess up the graph"""
    if not keys:
        return keys
        
    keys = sorted(keys)
    
    # If we have a reasonable number of keys, check if they follow a sequence
    if len(keys) > 3:
        # Calculate the median difference between consecutive keys
        diffs = [keys[i+1] - keys[i] for i in range(len(keys)-1)]
        median_diff = np.median(diffs)
        
        if median_diff > 0:
            # Find keys that have abnormally large gaps
            filtered_keys = [keys[0]]  # Always include the first key
            for i in range(1, len(keys)):
                # If this key is within a reasonable distance of the previous one
                if keys[i] - filtered_keys[-1] <= 3 * median_diff:
                    filtered_keys.append(keys[i])
                else:
                    # Only include the gap key if at least 50% of dataframes have it
                    # (This logic would need to be implemented separately)
                    pass
                    
            return filtered_keys
    
    # Default case: if we can't reasonably filter, return all keys
    # But exclude any extreme outliers (e.g., keys that are 10x larger than the median)
    if len(keys) > 5:
        median_key = keys[len(keys) // 2]
        max_reasonable_key = 5 * median_key
        return [k for k in keys if k <= max_reasonable_key]
    
    return keys


def plot(
    means,
    stdevs,
    mins,
    maxs,
    title,
    label,
    loc,
    xlabel="Training Epochs",
    ylabel="Top-1 Test Accuracy (%)",
    use_log=False,
):
    plt.title(title)
    plt.xlabel(xlabel)
    
    # Get keys and values as arrays
    x_keys = np.array(list(means.keys()))
    y_vals = np.array(list(means.values()))
    err = np.array([stdevs.get(k, float('nan')) for k in x_keys])
    
    # Apply logarithm transformation for loss plots if requested
    if use_log and "Loss" in title:
        # For log scale, filter out non-positive values first
        positive_mask = y_vals > 0
        
        # Create log-transformed values
        log_y_vals = np.full_like(y_vals, np.nan)
        log_y_vals[positive_mask] = np.log10(y_vals[positive_mask])
        
        # For error bars in log space, we need to transform the confidence bounds
        # Calculate lower and upper bounds in original space
        lower_bounds = y_vals - err
        upper_bounds = y_vals + err
        
        # Transform bounds to log space (only for positive values)
        log_lower = np.full_like(lower_bounds, np.nan)
        log_upper = np.full_like(upper_bounds, np.nan)
        
        # Only transform positive bounds
        positive_lower_mask = lower_bounds > 0
        positive_upper_mask = upper_bounds > 0
        
        log_lower[positive_lower_mask] = np.log10(lower_bounds[positive_lower_mask])
        log_upper[positive_upper_mask] = np.log10(upper_bounds[positive_upper_mask])
        
        # Calculate new error bars in log space
        # Use asymmetric error bars if bounds are different
        lower_err = log_y_vals - log_lower
        upper_err = log_upper - log_y_vals
        
        # Use the larger of the two errors for symmetric error bars
        # or handle NaN values gracefully
        err = np.nanmax([lower_err, upper_err], axis=0)
        
        # If we have NaN values, try to use at least one side
        nan_mask = np.isnan(err)
        err[nan_mask] = np.nanmax([lower_err[nan_mask], upper_err[nan_mask]], axis=0)
        
        # Update values
        y_vals = log_y_vals
        
        # Update ylabel to indicate logarithmic scale
        if "Cross Entropy Loss" in ylabel:
            ylabel = "log10(Cross Entropy Loss)"
        else:
            ylabel = "log10(Loss)"
    
    # Sort by x values to ensure correct plotting order
    sort_idx = np.argsort(x_keys)
    x_axis = x_keys[sort_idx]
    y_axis = y_vals[sort_idx]
    err = err[sort_idx]
    
    if "Muffliato" in label:
        x_axis = x_axis // MUFFLIATO_ROUNDS
    
    # Create masks for NaN values
    mask = ~np.isnan(y_axis)
    
    # Only plot non-NaN values
    if np.any(mask):
        plt.plot(x_axis[mask], y_axis[mask], label=label)
        plt.ylabel(ylabel)
        
        # Only fill between for non-NaN values where error is also valid
        err_mask = mask & ~np.isnan(err)
        if np.any(err_mask):
            lower_bound = y_axis - err
            upper_bound = y_axis + err
            
            plt.fill_between(
                x_axis[err_mask], 
                lower_bound[err_mask], 
                upper_bound[err_mask], 
                alpha=0.4
            )
    else:
        plt.plot([], [], label=f"{label} (all NaN)")
        
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc=loc)


def replace_dict_key(d_org: dict, d_other: dict):
    result = {}
    for x, y in d_org.items():
        result[d_other[x]] = y
    return result


def create_list_of_metrics(results, metric):
    return [x[metric][x[metric].notna()] for x in results if metric in x]

def get_min_max_test_acc(results):
    """Get the minimum of the maximum test accuracy across all dataframes, handling NaNs"""
    assert 'test_acc' in results[0].columns
    min_of_maxes_acc = float('inf')
    min_of_maxes_df_idx = 0
    min_of_maxes_row_idx = 0
    found_valid = False
    
    for df_idx, df in enumerate(results):
        # Filter out NaN values
        valid_df = df['test_acc'].dropna()
        
        if not valid_df.empty:
            max_acc = valid_df.max()
            max_acc_row_idx = valid_df.idxmax()
            
            if not found_valid or max_acc < min_of_maxes_acc:
                min_of_maxes_acc = max_acc
                min_of_maxes_df_idx = df_idx
                min_of_maxes_row_idx = max_acc_row_idx
                found_valid = True
    
    if not found_valid:
        return None, float('nan')  # Return appropriate values if all data is NaN
        
    return min_of_maxes_row_idx, min_of_maxes_acc


def extract_node_ids(filepaths):
    """Extract node IDs from CSV filenames"""
    node_ids = []
    for filepath in filepaths:
        # Filename format is "0_results.csv" where 0 is the node ID
        filename = os.path.basename(filepath)
        try:
            if '_results.csv' in filename:
                # Extract the part before "_results.csv"
                node_id = int(filename.split('_results.csv')[0])
                node_ids.append(node_id)
            else:
                node_ids.append(None)
        except (ValueError, IndexError):
            node_ids.append(None)
    
    return node_ids


def extract_corruption_metrics(folder_path):
    """
    Extract and aggregate max adversarial proportion from corruption metrics JSON files.
    Returns a dictionary with iteration -> sum of max_adversarial_proportion values.
    """
    aggregated_metrics = {}
    
    # Search for corruption_metrics_*.json files in all machine subfolders
    machine_folders = os.listdir(folder_path)
    for machine_folder in machine_folders:
        mf_path = os.path.join(folder_path, machine_folder)
        if not os.path.isdir(mf_path):
            continue
            
        files = os.listdir(mf_path)
        corruption_files = [f for f in files if f.startswith("corruption_metrics_") and f.endswith(".json")]
        
        for corruption_file in corruption_files:
            filepath = os.path.join(mf_path, corruption_file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle both old and new format
                if isinstance(data, dict):
                    if "adversarial_influence" in data:
                        # Old format - single entry
                        max_adv_prop = data["adversarial_influence"]["max_adversarial_proportion"]
                        current_round = data["current_round"]
                        
                        if current_round not in aggregated_metrics:
                            aggregated_metrics[current_round] = 0.0
                        aggregated_metrics[current_round] += max_adv_prop
                        
                    else:
                        # New format - iteration -> value mapping
                        for iteration_str, value in data.items():
                            try:
                                iteration = int(iteration_str)
                                if iteration not in aggregated_metrics:
                                    aggregated_metrics[iteration] = 0.0
                                aggregated_metrics[iteration] += value
                            except (ValueError, TypeError):
                                continue
                                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not read corruption metrics from {filepath}: {e}")
                continue
    
    return aggregated_metrics


def plot_results(results_path, config_path=None):
    folders = os.listdir(results_path)
    folders.sort()
    
    # Extract adversarial nodes from config if provided
    adversarial_nodes = []
    if config_path:
        adversarial_nodes = extract_adversarial_nodes(config_path)
        print(f"Extracted adversarial nodes from config: {adversarial_nodes}")
    
    print("Reading folders from: ", results_path)
    print("Folders: ", folders)
    if adversarial_nodes:
        print(f"Excluding adversarial nodes from metric computation: {adversarial_nodes}")
    
    bytes_means, bytes_stdevs = {}, {}
    meta_means, meta_stdevs = {}, {}
    data_means, data_stdevs = {}, {}
    
    for folder in folders:
        folder_path = Path(os.path.join(results_path, folder))
        if not folder_path.is_dir() or "weights" == folder_path.name:
            continue
            
        results = []
        filepaths = []  # Track filepaths to extract node IDs
        
        machine_folders = os.listdir(folder_path)
        for machine_folder in machine_folders:
            mf_path = os.path.join(folder_path, machine_folder)
            if not os.path.isdir(mf_path):
                continue
                
            files = os.listdir(mf_path)
            files = [f for f in files if f.endswith("_results.csv")]
            
            for f in files:
                filepath = os.path.join(mf_path, f)
                results.append(pd.read_csv(filepath, index_col=0))
                filepaths.append(filepath)

        # Extract node IDs from filenames
        node_ids = extract_node_ids(filepaths)
        
        # Extract and plot adversarial proportion metrics
        adversarial_metrics = extract_corruption_metrics(folder_path)
        if adversarial_metrics:
            print(f"Extracted adversarial metrics for {folder}: {len(adversarial_metrics)} iterations")
            plot_adversarial_proportion(adversarial_metrics, folder)
        else:
            print(f"No adversarial metrics found for {folder}")
        
        # Plot normal statistics (all nodes)
        plt.figure(1)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "train_loss")
        )
        plot(means, stdevs, mins, maxs, "Training Loss", folder, "upper right", use_log=True)
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(os.path.join(results_path, f"{folder}_train_loss.csv"), index_label="rounds")

        plt.figure(2)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "test_loss")
        )
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Convergence (Test Loss)",
            folder,
            "upper right",
            ylabel="Cross Entropy Loss",
            use_log=True
        )
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(os.path.join(results_path, f"{folder}_test_loss.csv"), index_label="rounds")

        plt.figure(3)
        min_of_maxes_row_idx, min_of_maxes_acc = get_min_max_test_acc(results)
        print(f"Minimum of maximum test accuracy: {min_of_maxes_acc} at row {min_of_maxes_row_idx} for folder {folder}")
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "test_acc")
        )
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Convergence (Test Accuracy)",
            folder,
            "lower right",
        )
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(os.path.join(results_path, f"{folder}_test_acc.csv"), index_label="rounds")
        
        # Only generate defender-only plots if we have adversarial nodes to exclude
        if adversarial_nodes:
            # Plot Training loss (defenders only)
            plt.figure(4)
            means, stdevs, mins, maxs, counts = get_stats(
                create_list_of_metrics(results, "train_loss"), adversarial_nodes, node_ids
            )
            plot(means, stdevs, mins, maxs, "Training Loss (Defenders Only)", folder, "upper right", use_log=True)
            df = pd.DataFrame(
                {
                    "mean": list(means.values()),
                    "std": list(stdevs.values()),
                    "nr_nodes": counts,
                },
                list(means.keys()),
                columns=["mean", "std", "nr_nodes"],
            )
            df.to_csv(os.path.join(results_path, f"{folder}_train_loss_defenders.csv"), index_label="rounds")
            
            # Plot Testing loss (defenders only)
            plt.figure(5)
            means, stdevs, mins, maxs, counts = get_stats(
                create_list_of_metrics(results, "test_loss"), adversarial_nodes, node_ids
            )
            plot(
                means,
                stdevs,
                mins,
                maxs,
                "Convergence (Test Loss - Defenders Only)",
                folder,
                "upper right",
                ylabel="Cross Entropy Loss",
                use_log=True
            )
            df = pd.DataFrame(
                {
                    "mean": list(means.values()),
                    "std": list(stdevs.values()),
                    "nr_nodes": counts,
                },
                list(means.keys()),
                columns=["mean", "std", "nr_nodes"],
            )
            df.to_csv(os.path.join(results_path, f"{folder}_test_loss_defenders.csv"), index_label="rounds")
            
            # Plot Testing Accuracy (defenders only)
            plt.figure(6)
            means, stdevs, mins, maxs, counts = get_stats(
                create_list_of_metrics(results, "test_acc"), adversarial_nodes, node_ids
            )
            plot(
                means,
                stdevs,
                mins,
                maxs,
                "Convergence (Test Accuracy - Defenders Only)",
                folder,
                "lower right",
            )
            df = pd.DataFrame(
                {
                    "mean": list(means.values()),
                    "std": list(stdevs.values()),
                    "nr_nodes": counts,
                },
                list(means.keys()),
                columns=["mean", "std", "nr_nodes"],
            )
            df.to_csv(os.path.join(results_path, f"{folder}_test_acc_defenders.csv"), index_label="rounds")

    # Save all figures
    plt.figure(1)
    plt.savefig(os.path.join(results_path, "train_loss.pdf"), dpi=600)
    plt.figure(2)
    plt.savefig(os.path.join(results_path, "test_loss.pdf"), dpi=600)
    plt.figure(3)
    plt.savefig(os.path.join(results_path, "test_acc.pdf"), dpi=600)
    plt.figure(7)
    plt.savefig(os.path.join(results_path, "adversarial_proportion.pdf"), dpi=600)
    
    if adversarial_nodes:
        plt.figure(4)
        plt.savefig(os.path.join(results_path, "train_loss_defenders.pdf"), dpi=600)
        plt.figure(5)
        plt.savefig(os.path.join(results_path, "test_loss_defenders.pdf"), dpi=600)
        plt.figure(6)
        plt.savefig(os.path.join(results_path, "test_acc_defenders.pdf"), dpi=600)


def plot_adversarial_proportion(adversarial_metrics, folder, figure_num=7):
    """
    Plot adversarial proportion values as a separate graph.
    """
    if not adversarial_metrics:
        print(f"No adversarial metrics data for {folder}")
        return
    
    plt.figure(figure_num)
    
    iterations = sorted(adversarial_metrics.keys())
    values = [adversarial_metrics[it] for it in iterations]
    
    plt.plot(iterations, values, label=f'{folder} - Sum of Max Adversarial Proportion', 
             marker='o', linewidth=2, markersize=4)
    
    plt.title('Sum of Max Adversarial Proportion Over Iterations')
    plt.xlabel('Training Iterations')
    plt.ylabel('Sum of Max Adversarial Proportion')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python plot_csv_acc_defenders.py <results_path> [config_path]")
        sys.exit(1)
        
    results_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    plot_results(results_path, config_path)