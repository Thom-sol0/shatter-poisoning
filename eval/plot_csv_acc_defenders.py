import os
import sys
from pathlib import Path
import configparser

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
    """Calculate statistics excluding adversarial nodes"""
    assert len(l) > 0
    mean_dict, stdev_dict, min_dict, max_dict, counts_dict = {}, {}, {}, {}, {}
    
    for key in l[0].index:
        if MAX_ITERATION is not None and key >= MAX_ITERATION:
            continue
            
        # Filter to include only defender nodes if node_ids are provided
        if node_ids and adversarial_nodes:
            all_nodes = [i[key] for i, node_id in zip(l, node_ids) if node_id not in adversarial_nodes]
        else:
            all_nodes = [i[key] for i in l]
            
        if not all_nodes:  # Skip if no defender nodes found
            continue
            
        all_nodes = np.array(all_nodes)
        mean = np.mean(all_nodes)
        std = np.std(all_nodes)
        min = np.min(all_nodes)
        max = np.max(all_nodes)
        count = np.count_nonzero(~np.isnan(all_nodes))
        
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min
        max_dict[int(key)] = max
        counts_dict[int(key)] = count
        
    return mean_dict, stdev_dict, min_dict, max_dict, counts_dict


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
):
    plt.title(title)
    plt.xlabel(xlabel)
    x_axis = np.array(list(means.keys()))
    if "Muffliato" in label:
        x_axis = x_axis // MUFFLIATO_ROUNDS
    y_axis = np.array(list(means.values()))
    err = np.array(list(stdevs.values()))
    plt.plot(x_axis, y_axis, label=label)
    plt.ylabel(ylabel)
    plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)
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
        
        # Plot normal statistics (all nodes)
        plt.figure(1)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "train_loss")
        )
        plot(means, stdevs, mins, maxs, "Training Loss", folder, "upper right")
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
            plot(means, stdevs, mins, maxs, "Training Loss (Defenders Only)", folder, "upper right")
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
    
    if adversarial_nodes:
        plt.figure(4)
        plt.savefig(os.path.join(results_path, "train_loss_defenders.pdf"), dpi=600)
        plt.figure(5)
        plt.savefig(os.path.join(results_path, "test_loss_defenders.pdf"), dpi=600)
        plt.figure(6)
        plt.savefig(os.path.join(results_path, "test_acc_defenders.pdf"), dpi=600)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python plot_csv_acc_defenders.py <results_path> [config_path]")
        sys.exit(1)
        
    results_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    plot_results(results_path, config_path)