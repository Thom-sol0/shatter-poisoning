from collections import defaultdict
import torch
import os
import pandas as pd
import numpy as np
from virtualNodes.sharing.VNodeSharingDefenseWeightTracker import VNodeSharingDefenseWeightTracker

class VNodeSharingDefenseNoneWithWeightTracking(VNodeSharingDefenseWeightTracker):
    """
    "None" defense implementation with weight tracking capabilities
    This class replicates VNodeSharingDefenseNone behavior but adds weight tracking for L2 norm analysis
    """

    def initialize_defense_data(self):
        """
        Initialize defense-specific data structures
        For "None" defense, we just need to track received models
        """
        self.defense_data = {
            "received_models": defaultdict(list),  # Maps node_id -> list of models received from that node
            "round_counts": defaultdict(int),      # Counts how many models we've received from each node this round
        }

    def _cleanup_defense_data(self):
        """
        Clean up defense-specific data structures
        """
        # Clear received models but maintain the structure
        self.defense_data["received_models"] = defaultdict(list)
        self.defense_data["round_counts"] = defaultdict(int)

    def defender_forward_averaging(self, data):
        """
        Forward averaging function for defenders that implements "None" defense
        (same as standard averaging with additional tracking)
        """
        if self.current_sum is None:
            self.current_weights = (
                torch.zeros(self.total_length, dtype=torch.float32, device=self.device) + 1
            )

            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            self.current_sum = torch.cat(tensors_to_cat, dim=0).to(self.device)

        iteration = data["iteration"]
        sender_node = data.get("vSource", "unknown")
        real_node_id = data.get("real_node", None)

        # Track neighbors for logging
        self.neighbor_list.append(real_node_id)

        # Clean up data
        for key in ["degree", "iteration", "CHANNEL", "real_node"]:
            if key in data:
                del data[key]

        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e

        # Track received model for defense mechanisms
        if real_node_id is not None:
            self.defense_data["received_models"][real_node_id].append({
                "weights": deserializedT.clone().cpu(),
                "indices": indices.clone().cpu()
            })
            self.defense_data["round_counts"][real_node_id] += 1

        # Sanitize NaN/Inf values in received weights
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )

        # Sanitize NaN/Inf values in current sum before update
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_before_update", 
                f"node_{self.uid}"
            )

        # Add received weights to current sum
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1

        # Sanitize NaN/Inf values in current sum after update
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_update", 
                f"node_{self.uid}"
            )

    def get_defended_model(self):
        """
        Get the defended model using the "None" defense (standard averaging)
        """
        # Safety check
        assert self.current_sum is not None
        assert self.current_weights is not None

        # Sanitize NaN/Inf values in final sum and weights
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_final", 
                f"node_{self.uid}"
            )

        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            self.current_weights = self._detect_and_sanitize_nan_inf(
                self.current_weights, 
                "current_weights_final", 
                f"node_{self.uid}",
                default_value=1.0
            )

        # Convert weights to float
        self.current_weights = self.current_weights.type(torch.float32)

        # Handle zero weights (avoid division by zero)
        zero_weights_mask = self.current_weights == 0
        if torch.any(zero_weights_mask):
            self.current_weights[zero_weights_mask] = 1.0

        # Calculate average (divide by count)
        self.current_weights = 1.0 / self.current_weights

        # Sanitize weights after division
        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            self.current_weights = self._detect_and_sanitize_nan_inf(
                self.current_weights, 
                "current_weights_after_division", 
                f"node_{self.uid}",
                default_value=1.0
            )

        # Apply weights to sum
        self.current_sum = self.current_sum * self.current_weights

        # Final sanity check
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_multiplication", 
                f"node_{self.uid}"
            )

        # Move to CPU and convert to model state dict
        self.current_sum = self.current_sum.cpu()
        
        # Get the final state dict
        return self._post_step(self.current_sum)
        
    def generate_weight_diff_report(self, other_experiment_id, output_file=None):
        """
        Generate a detailed report comparing weights between experiments
        
        Parameters:
        -----------
        other_experiment_id : str
            ID of another experiment to compare with
        output_file : str, optional
            Path to save the report (defaults to log directory)
            
        Returns:
        --------
        str : Path to the report file
        """
        if output_file is None:
            output_file = os.path.join(
                self.log_dir, 
                f"weight_diff_report_{self.experiment_id}_vs_{other_experiment_id}.md"
            )
            
        # Compute L2 norm differences
        csv_file = self.export_weight_diff_analysis(other_experiment_id)
        df = pd.read_csv(csv_file)
        
        # Generate report
        with open(output_file, "w") as f:
            f.write(f"# Weight Difference Report\n\n")
            f.write(f"- Experiment 1: {self.experiment_id}\n")
            f.write(f"- Experiment 2: {other_experiment_id}\n")
            f.write(f"- Node: {self.uid}\n\n")
            
            f.write("## L2 Norm Differences\n\n")
            
            # Add a plot description
            f.write("The following table shows the L2 norm difference between model weights\n")
            f.write("at different communication rounds:\n\n")
            
            # Convert DataFrame to markdown table
            f.write(df[["round", "total"]].to_markdown(index=False))
            
            f.write("\n\n## Analysis\n\n")
            
            # Basic statistics
            avg_diff = df["total"].mean()
            max_diff = df["total"].max()
            min_diff = df["total"].min()
            
            f.write(f"- Average L2 norm difference: {avg_diff:.6f}\n")
            f.write(f"- Maximum L2 norm difference: {max_diff:.6f} (round {df.loc[df['total'].idxmax(), 'round']})\n")
            f.write(f"- Minimum L2 norm difference: {min_diff:.6f} (round {df.loc[df['total'].idxmin(), 'round']})\n\n")
            
            # Growth rate analysis
            if len(df) > 1:
                first_round = df.iloc[0]["round"]
                last_round = df.iloc[-1]["round"]
                first_norm = df.iloc[0]["total"]
                last_norm = df.iloc[-1]["total"]
                
                growth_rate = (last_norm - first_norm) / (last_round - first_round)
                f.write(f"- Average growth rate: {growth_rate:.6f} per round\n")
                
            f.write("\n\n*This report was generated automatically*\n")
            
        return output_file
