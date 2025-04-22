#!/usr/bin/env python3

import argparse
import configparser
import os
import sys

def create_poisoning_config(base_config_path, output_config_path, 
                           attack_type='zero', poison_strength=1.0,
                           poison_probability=1.0, node_ids=None,
                           log_poisoning_metrics=True):
    """
    Create a poisoned version of a config file by changing the sharing class.
    
    Parameters
    ----------
    base_config_path : str
        Path to the base config file
    output_config_path : str
        Path to write the modified config
    attack_type : str
        Type of poisoning attack ('zero', 'flip', 'noise', 'scale')
    poison_strength : float
        Strength of the poisoning attack
    poison_probability : float
        Probability of poisoning a message
    node_ids : list of int
        List of node IDs to poison (None = all nodes become adversarial)
    log_poisoning_metrics : bool
        Whether to log poisoning metrics
    """
    
    config = configparser.ConfigParser()
    config.read(base_config_path)
    
    # Change the sharing section to use our poisoning class
    if 'SHARING' not in config:
        raise ValueError("Config file missing SHARING section")
    
    # Save original sharing class for non-poisoning nodes
    original_sharing_package = config['SHARING']['sharing_package']
    original_sharing_class = config['SHARING']['sharing_class']
    
    # Update sharing config to use poisoning class
    config['SHARING']['sharing_package'] = 'virtualNodes.sharing.VNodeSharingPoison'
    config['SHARING']['sharing_class'] = 'VNodeSharingPoison'
    
    # Add poisoning parameters
    config['SHARING']['attack_type'] = attack_type
    config['SHARING']['poison_strength'] = str(poison_strength)
    config['SHARING']['poison_probability'] = str(poison_probability)
    if node_ids:
        config['SHARING']['targeted_nodes'] = ','.join(map(str, node_ids))
    config['SHARING']['log_poisoning_metrics'] = str(log_poisoning_metrics).lower()
    
    # Write the new config
    with open(output_config_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"Created poisoned config at {output_config_path}")
    print(f"Attack type: {attack_type}")
    print(f"Poison strength: {poison_strength}")
    print(f"Poison probability: {poison_probability}")

def main():
    parser = argparse.ArgumentParser(description='Create a config file for poisoning experiments')
    parser.add_argument('--base-config', required=True, help='Base config file path')
    parser.add_argument('--output-config', required=True, help='Output config file path')
    parser.add_argument('--attack-type', default='zero', choices=['zero', 'flip', 'noise', 'scale'], 
                        help='Type of poisoning attack')
    parser.add_argument('--poison-strength', type=float, default=1.0, 
                        help='Strength of the poisoning attack')
    parser.add_argument('--poison-probability', type=float, default=1.0, 
                        help='Probability of poisoning a message')
    parser.add_argument('--node-ids', type=int, nargs='*', 
                        help='Node IDs to poison (default: all nodes)')
    parser.add_argument('--log-metrics', action='store_true', default=True, 
                        help='Log poisoning metrics')
    
    args = parser.parse_args()
    
    create_poisoning_config(
        args.base_config,
        args.output_config,
        args.attack_type,
        args.poison_strength,
        args.poison_probability,
        args.node_ids,
        args.log_metrics
    )

if __name__ == "__main__":
    main()