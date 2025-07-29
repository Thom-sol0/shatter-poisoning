import copy
import logging
import os
import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from virtualNodes.sharing.VNodeSharingDefenseBase import VNodeSharingDefenseBase

class VNodeSharingDefenseWeightTracker(VNodeSharingDefenseBase):
    """
    Extended defense base class that tracks model weights at regular intervals
    for later analysis and comparison between different runs.
    """
    
    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
        attack_type='zero',
        adversarial_nodes=None,
        poison_after=None,
        save_interval=10,  # Save weights every N communication rounds
        experiment_id=None,  # Unique identifier for this experiment run
    ):
        """
        Constructor for weight-tracking defense class
        
        Parameters:
        -----------
        save_interval : int
            Interval (in communication rounds) at which to save model weights
        experiment_id : str
            Unique identifier for this experiment run to differentiate between runs
        """
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            compress=compress,
            compression_package=compression_package,
            compression_class=compression_class,
            float_precision=float_precision,
            attack_type=attack_type,
            adversarial_nodes=adversarial_nodes,
            poison_after=poison_after,
        )
        
        # Weight tracking parameters
        self.save_interval = save_interval
        
        # Create unique experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attacker_count = len(self.adversarial_nodes) if self.adversarial_nodes else 0
            self.experiment_id = f"exp_{timestamp}_atk{attacker_count}"
        else:
            self.experiment_id = experiment_id
            
        # Create directory for saved weights
        self.weights_dir = os.path.join(self.log_dir, "saved_weights", self.experiment_id)
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Save experiment metadata
        self._save_experiment_metadata()
        
        # Initialize weight history dictionary
        self.weight_history = {}

    def _save_experiment_metadata(self):
        """
        Save metadata about this experiment for reference
        """
        metadata = {
            "experiment_id": self.experiment_id,
            "node_id": self.uid,
            "is_adversary": self.uid in self.adversarial_nodes,
            "adversarial_nodes": self.adversarial_nodes,
            "attack_type": self.attack_type,
            "poison_after": self.poison_after,
            "save_interval": self.save_interval,
            "timestamp": datetime.now().isoformat(),
        }
        
        metadata_file = os.path.join(self.weights_dir, f"metadata_node{self.uid}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_model_weights(self):
        """
        Save the current model weights to a file
        """
        if self.communication_round % self.save_interval == 0:
            # Create a deep copy to prevent modification
            state_dict = copy.deepcopy(self.model.state_dict())
            
            # Convert model weights to CPU for consistent storage
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            
            # Save to file
            weights_file = os.path.join(
                self.weights_dir, 
                f"node{self.uid}_round{self.communication_round}.pt"
            )
            
            torch.save(cpu_state_dict, weights_file)
            
            # Also store in memory (for quick access during the run)
            self.weight_history[self.communication_round] = cpu_state_dict
            
            logging.info(f"Node {self.uid}: Saved weights at round {self.communication_round}")
            
    def compute_weight_diff_l2_norm(self, other_weights_dir, round_num):
        """
        Compute L2 norm of weight difference between this run and another run at a specific round
        
        Parameters:
        -----------
        other_weights_dir : str
            Path to another experiment's saved weights directory
        round_num : int
            Communication round to compare
            
        Returns:
        --------
        dict : Dictionary of L2 norms per layer and total L2 norm
        """
        current_weights_file = os.path.join(
            self.weights_dir, 
            f"node{self.uid}_round{round_num}.pt"
        )
        
        other_weights_file = os.path.join(
            other_weights_dir, 
            f"node{self.uid}_round{round_num}.pt"
        )
        
        if not os.path.exists(current_weights_file) or not os.path.exists(other_weights_file):
            return {"error": "Weight files not found for comparison"}
        
        current_state = torch.load(current_weights_file)
        other_state = torch.load(other_weights_file)
        
        l2_norms = {}
        total_squared_diff = 0.0
        
        # Compute L2 norm for each layer
        for key in current_state:
            if key in other_state:
                diff = current_state[key] - other_state[key]
                layer_norm = torch.norm(diff).item()
                l2_norms[key] = layer_norm
                total_squared_diff += layer_norm ** 2
        
        # Compute total L2 norm across all layers
        l2_norms["total"] = np.sqrt(total_squared_diff)
        
        return l2_norms
            
    def export_weight_diff_analysis(self, other_experiment_id, interval=None):
        """
        Export a CSV with L2 norm differences between this experiment and another one
        
        Parameters:
        -----------
        other_experiment_id : str
            ID of another experiment to compare with
        interval : int or None
            If provided, only analyze differences at this interval
            
        Returns:
        --------
        str : Path to the CSV file with results
        """
        other_weights_dir = os.path.join(
            os.path.dirname(self.weights_dir),
            other_experiment_id
        )
        
        if not os.path.exists(other_weights_dir):
            raise ValueError(f"Experiment directory not found: {other_weights_dir}")
            
        # Determine rounds to analyze
        if interval is None:
            interval = self.save_interval
            
        max_round = max([
            int(f.name.split("_round")[1].split(".pt")[0])
            for f in Path(self.weights_dir).glob(f"node{self.uid}_round*.pt")
        ])
        
        rounds_to_analyze = list(range(0, max_round + 1, interval))
        
        # Compute differences
        results = []
        for round_num in rounds_to_analyze:
            weights_file = os.path.join(
                self.weights_dir, 
                f"node{self.uid}_round{round_num}.pt"
            )
            
            if not os.path.exists(weights_file):
                continue
                
            diff_norms = self.compute_weight_diff_l2_norm(other_weights_dir, round_num)
            if "error" not in diff_norms:
                diff_norms["round"] = round_num
                results.append(diff_norms)
        
        # Export to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        
        output_file = os.path.join(
            self.log_dir,
            f"weight_diff_{self.experiment_id}_vs_{other_experiment_id}_node{self.uid}.csv"
        )
        
        df.to_csv(output_file, index=False)
        
        # Also create a summary with just the total norm
        summary_file = os.path.join(
            self.log_dir,
            f"weight_diff_summary_{self.experiment_id}_vs_{other_experiment_id}_node{self.uid}.csv"
        )
        
        df[["round", "total"]].to_csv(summary_file, index=False)
        
        return output_file

    def finish_forward_averaging(self, peer_deques):
        """
        Override finish_forward_averaging to include weight saving
        """
        # Call the parent's implementation
        super().finish_forward_averaging(peer_deques)
        
        # Save weights at the specified interval
        self._save_model_weights()
    
    # Required abstract method implementations
    # These should be implemented in your concrete defense classes
    
    def defender_forward_averaging(self, data):
        raise NotImplementedError("This class must be extended with a specific defense mechanism")
        
    def get_defended_model(self):
        raise NotImplementedError("This class must be extended with a specific defense mechanism")
        
    def initialize_defense_data(self):
        raise NotImplementedError("This class must be extended with a specific defense mechanism")
        
    def _cleanup_defense_data(self):
        raise NotImplementedError("This class must be extended with a specific defense mechanism")
