import copy
import logging
import os
import torch
import numpy as np
import json
from virtualNodes.sharing.VNodeSharingRandom import VNodeSharing
from collections import defaultdict
import warnings

class VNodeSharingPoison(VNodeSharing):
    """
    Poisoned model sharing class with clipping-based defense mechanism.
    
    This class implements a gradient clipping defense that identifies and clips
    outlier gradients based on statistical measures to prevent large deviations
    from reasonable parameter updates.
    """

    def _parse_adversarial_nodes(self, adversarial_nodes):
        """
        Parse the adversarial nodes input to ensure it's a list of integers.
        Handles string input, single integers, and lists.
        """
        if isinstance(adversarial_nodes, str) and adversarial_nodes:
            # Parse string representation
            return [int(node_id.strip()) for node_id in adversarial_nodes.split(',') if node_id.strip()]
        elif adversarial_nodes is None:
            # No adversarial nodes
            return []
        elif not isinstance(adversarial_nodes, list):
            # Single integer or other non-list type
            return [adversarial_nodes]
        else:
            # Already a list
            return adversarial_nodes
    
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
        adversarial_nodes=None,   # List of uids of adversarial nodes
        poison_after=None,
        clipping_factor=2.0,     # How many std deviations to allow before clipping
        percentile_threshold=75,  # Percentile for determining reasonable range
        adaptive_clipping=True,   # Whether to adapt clipping bounds over time
        # log_poisoning_metrics=True,
    ):
        """
        Constructor for clipping-based defense class
        
        Parameters
        ----------
        rank, machine_id, etc. : same as parent class
        attack_type : str
            Poisoning strategy ('zero', 'flip', 'noise', 'scale')
        adversarial_nodes : list
            List of node IDs that will perform the attack
        poison_after : int
            Number of rounds between poisonings
        clipping_factor : float
            Number of standard deviations to allow before clipping
        percentile_threshold : int
            Percentile threshold for determining reasonable parameter ranges
        adaptive_clipping : bool
            Whether to adapt clipping bounds based on historical data
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
            float_precision=float_precision
        )
        
        self.attack_type = attack_type
        self.poison_after = int(poison_after) if poison_after is not None else 1
        self.adversarial_nodes = self._parse_adversarial_nodes(adversarial_nodes)

        # Clipping defense parameters
        self.clipping_factor = clipping_factor
        self.percentile_threshold = percentile_threshold
        self.adaptive_clipping = adaptive_clipping
        
        # Data structures for collecting parameter values for clipping analysis
        self.param_values = None
        self.param_adversarial_sources = None
        
        # Historical statistics for adaptive clipping
        self.param_history = defaultdict(list)  # Store recent parameter values for each position
        self.history_window = 10  # Number of rounds to keep for statistics
        
        # Clipping statistics
        self.clipped_weights_count = 0
        self.total_weights_processed = 0
        self.clipping_bounds_history = {}
        
        # Diagnostic tracking
        self.nan_inf_detection_count = 0
        self.corrupted_weights_received = 0
        self.corrupted_weights_rejected = 0
        self.max_adversarial_proportion = 0.0

    def _detect_and_sanitize_nan_inf(self, tensor, tensor_name="tensor", sender_node="unknown", default_value=0.0):
        """
        Detect and sanitize NaN/Inf values in tensors.
        """
        if tensor is None:
            return torch.zeros_like(tensor) if tensor is not None else None
        
        # Check for NaN values
        nan_mask = torch.isnan(tensor)
        nan_count = torch.sum(nan_mask).item()
        
        # Check for Inf values
        inf_mask = torch.isinf(tensor)
        inf_count = torch.sum(inf_mask).item()
        
        # Check for extremely large values that might cause overflow
        large_mask = torch.abs(tensor) > 1e10
        large_count = torch.sum(large_mask).item()
        
        total_corrupted = nan_count + inf_count
        
        if total_corrupted > 0:
            self.nan_inf_detection_count += 1
            self.corrupted_weights_received += total_corrupted
            
            # Sanitize the tensor
            sanitized_tensor = tensor.clone()
            
            # Replace NaN with default value
            if nan_count > 0:
                sanitized_tensor[nan_mask] = default_value
            
            # Replace Inf with clamped values
            if inf_count > 0:
                pos_inf_mask = torch.isposinf(tensor)
                neg_inf_mask = torch.isneginf(tensor)
                sanitized_tensor[pos_inf_mask] = 1e6
                sanitized_tensor[neg_inf_mask] = -1e6
            
            # Clamp extremely large values
            if large_count > 0:
                sanitized_tensor = torch.clamp(sanitized_tensor, min=-1e8, max=1e8)
            
            self.corrupted_weights_rejected += total_corrupted
            return sanitized_tensor
        
        return tensor

    def _validate_model_state(self, state_dict, source="unknown"):
        """
        Validate and sanitize model state dictionary for NaN/Inf values.
        """
        corrupted_layers = []
        total_corrupted = 0
        
        for layer_name, tensor in state_dict.items():
            if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                corrupted_layers.append(layer_name)
                nan_count = torch.sum(torch.isnan(tensor)).item()
                inf_count = torch.sum(torch.isinf(tensor)).item()
                total_corrupted += nan_count + inf_count
                
                # Sanitize this layer
                sanitized_tensor = self._detect_and_sanitize_nan_inf(tensor, f"layer_{layer_name}", source)
                state_dict[layer_name] = sanitized_tensor
        
        return state_dict, len(corrupted_layers) > 0

    def _compute_clipping_bounds(self, param_values_list):
        """
        Compute clipping bounds for a list of parameter values using statistical measures.
        
        Parameters
        ----------
        param_values_list : list
            List of parameter values for a specific weight position
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound) for clipping
        """
        if len(param_values_list) < 3:
            # Not enough data for meaningful statistics, use simple bounds
            return -1e6, 1e6
        
        values = np.array(param_values_list)
        
        # Method 1: Use standard deviation-based clipping
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Prevent division by zero
        if std_val < 1e-10:
            std_val = 1e-6
        
        std_lower = mean_val - self.clipping_factor * std_val
        std_upper = mean_val + self.clipping_factor * std_val
        
        # Method 2: Use percentile-based clipping for robustness
        lower_percentile = (100 - self.percentile_threshold) / 2
        upper_percentile = 100 - lower_percentile
        
        percentile_lower = np.percentile(values, lower_percentile)
        percentile_upper = np.percentile(values, upper_percentile)
        
        # Method 3: Use IQR-based outlier detection
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        
        # Use a factor of 1.5 * IQR for outlier detection (standard approach)
        iqr_lower = q25 - 1.5 * iqr
        iqr_upper = q75 + 1.5 * iqr
        
        # Combine all methods - use the most restrictive bounds
        final_lower = max(std_lower, percentile_lower, iqr_lower)
        final_upper = min(std_upper, percentile_upper, iqr_upper)
        
        # Ensure bounds are reasonable and not too restrictive
        if final_upper - final_lower < 1e-6:
            # Bounds are too tight, fallback to std-based with larger factor
            final_lower = mean_val - 3.0 * std_val
            final_upper = mean_val + 3.0 * std_val
        
        return final_lower, final_upper

    def _update_parameter_history(self, param_idx, value):
        """
        Update historical parameter values for adaptive clipping.
        
        Parameters
        ----------
        param_idx : int
            Parameter index
        value : float
            Parameter value to add to history
        """
        if self.adaptive_clipping:
            self.param_history[param_idx].append(value)
            
            # Keep only recent history
            if len(self.param_history[param_idx]) > self.history_window:
                self.param_history[param_idx].pop(0)

    def _get_adaptive_clipping_bounds(self, param_idx, current_values):
        """
        Get adaptive clipping bounds considering historical data.
        
        Parameters
        ----------
        param_idx : int
            Parameter index
        current_values : list
            Current round parameter values
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound) for clipping
        """
        if not self.adaptive_clipping or param_idx not in self.param_history:
            return self._compute_clipping_bounds(current_values)
        
        # Combine historical and current values
        historical_values = self.param_history[param_idx]
        combined_values = historical_values + current_values
        
        return self._compute_clipping_bounds(combined_values)

    def _apply_model_poisoning(self):
        """
        Apply poisoning strategy for adversarial nodes
        """
        # Check if we should poison in this round (based on poison_after)
        if self.poison_after > 1 and self.communication_round % self.poison_after != 0:
            # Just apply the averaged model when poisoning is skipped
            mean_state_dict = self._post_step(self.current_sum)
            mean_state_dict, was_corrupted = self._validate_model_state(mean_state_dict, f"mean_averaging_node_{self.uid}")
            self.model.load_state_dict(mean_state_dict)
            return
    
        if self.attack_type == 'zero':
            # Zero out all parameters
            self.current_sum.zero_()
            zero_state_dict = self._post_step(self.current_sum)
            zero_state_dict, was_corrupted = self._validate_model_state(zero_state_dict, f"zero_attack_node_{self.uid}")
            self.model.load_state_dict(zero_state_dict)
        
        elif self.attack_type == 'flip_grad':
            # Mirror the gradient update to move away from consensus
            mean_state_dict = self._post_step(self.current_sum)
            init_state_dict = copy.deepcopy(self.model.state_dict())
            
            # Validate input states
            mean_state_dict, mean_corrupted = self._validate_model_state(mean_state_dict, f"mean_state_node_{self.uid}")
            init_state_dict, init_corrupted = self._validate_model_state(init_state_dict, f"init_state_node_{self.uid}")
            
            final_state_dict = {}
            
            for key in init_state_dict:
                # Perform flip operation: new = 2 * init - mean
                try:
                    flipped_update = 2 * init_state_dict[key] - mean_state_dict[key]
                    
                    # Check for NaN/Inf in the result
                    if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                        # Try sanitized version
                        flipped_update = self._detect_and_sanitize_nan_inf(
                            flipped_update, 
                            f"flip_grad_{key}", 
                            f"node_{self.uid}",
                            default_value=0.0
                        )
                        
                        # If still problematic, fallback to mean state
                        if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                            final_state_dict[key] = mean_state_dict[key]
                        else:
                            final_state_dict[key] = flipped_update
                    else:
                        # Check for extremely large values that might become problematic
                        if torch.max(torch.abs(flipped_update)) > 1e8:
                            flipped_update = torch.clamp(flipped_update, min=-1e8, max=1e8)
                        final_state_dict[key] = flipped_update
                        
                except Exception as e:
                    final_state_dict[key] = mean_state_dict[key]
            
            # Final validation before loading
            final_state_dict, final_corrupted = self._validate_model_state(final_state_dict, f"final_flip_grad_node_{self.uid}")
            self.model.load_state_dict(final_state_dict)
        
        else:
            # Use normal averaged model for unknown attack types
            mean_state_dict = self._post_step(self.current_sum)
            mean_state_dict, was_corrupted = self._validate_model_state(mean_state_dict, f"fallback_node_{self.uid}")
            self.model.load_state_dict(mean_state_dict)

    def get_data_to_send(self, vnodes_per_node=1, degree=None, sparsity=0.0):
        """
        Get data to send to neighbors, including real node ID for proper tracking.
        """
        self._pre_step()
        data_list = self.serialized_models(
            vnodes_per_node=vnodes_per_node, sparsity=sparsity
        )
        # Add real node ID to each data packet for proper adversarial tracking
        for data in data_list:
            data["real_node"] = self.uid
        return data_list

    def _apply_defense(self):
        """
        Apply defense mechanisms against poisoning attacks
        """
        # This will be implemented in the clipping-based defense mechanism
        pass

    def forward_averaging(self, data):
        if self.uid in self.adversarial_nodes:
            self.adversarial_forward_averaging(data)
        else:
            self.defender_forward_averaging(data)

    def adversarial_forward_averaging(self, data):
        """
        Computes the sum for the average in a state based manner for adversarial nodes.
        """
        if self.current_sum == None:
            # First time take model of self
            self.current_weights = (
                torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
                + 1
            )

            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            self.current_sum = torch.cat(tensors_to_cat, dim=0).to(self.device)

        iteration = data["iteration"]
        sender_node = data.get("vSource", "unknown")
        real_node_id = data.get("real_node", None)
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        if "real_node" in data:
            del data["real_node"]
        
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        
        # Validate received weights for NaN/Inf before adding them
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )
        
        # Also validate current_sum before updating
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_before_update", 
                f"node_{self.uid}"
            )
        
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1
        
        # Validate current_sum after update
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_update", 
                f"node_{self.uid}"
            )

    def defender_forward_averaging(self, data):
        """
        Computes and collects parameter values for clipping-based aggregation.
        """
        if self.param_values is None:
            # First time, initialize param_values dictionary
            self.param_values = defaultdict(list)
            # Also initialize adversarial sources tracking
            self.param_adversarial_sources = defaultdict(list)
        
            # Add own model weights as the first entry
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            own_weights = torch.cat(tensors_to_cat, dim=0).to(self.device)
        
            # Add own weights to param_values and mark as self (not adversarial for defense purposes)
            for i in range(len(own_weights)):
                self.param_values[i].append(own_weights[i].item())
                # Mark own weights as coming from self (not adversarial for counting purposes)
                self.param_adversarial_sources[i].append((self.uid, False))

        # Process received model data
        iteration = data["iteration"]
        sender_node = data.get("real_node", data.get("vSource", "unknown"))  # Get sender info
        real_node_id = data.get("real_node", None)
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        
        # Remove real_node from data if it exists (after extracting it)
        if "real_node" in data:
            del data["real_node"]
    
        # Deserialize received model data
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        
        # Validate received weights for NaN/Inf before processing
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )
    
        # Add the received weights to the param_values dictionary
        # and track whether they come from adversarial nodes
        sender_is_adversarial = sender_node in self.adversarial_nodes
        for idx, value in zip(indices.tolist(), deserializedT.tolist()):
            # Additional check for individual values
            if np.isnan(value) or np.isinf(value):
                value = 0.0  # Replace with safe default
            self.param_values[idx].append(value)
            # Track the source and whether it's adversarial
            self.param_adversarial_sources[idx].append((sender_node, sender_is_adversarial))
        
    def adversarial_finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging for adversarial nodes.
        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        # Validate current_sum and current_weights before final processing
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
                default_value=1.0  # For weights, use 1.0 as default
            )

        self.current_weights = self.current_weights.type(torch.float32)
        
        # Prevent division by zero and validate weights
        zero_weights_mask = self.current_weights == 0
        if torch.any(zero_weights_mask):
            self.current_weights[zero_weights_mask] = 1.0
        
        self.current_weights = 1.0 / self.current_weights
        
        # Check for NaN/Inf after division
        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            self.current_weights = self._detect_and_sanitize_nan_inf(
                self.current_weights, 
                "current_weights_after_division", 
                f"node_{self.uid}",
                default_value=1.0
            )
        
        self.current_sum = self.current_sum * self.current_weights
        
        # Final validation of current_sum
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_multiplication", 
                f"node_{self.uid}"
            )
        
        self.current_sum = self.current_sum.cpu()

        self._apply_model_poisoning()
        
        self.communication_round += 1
        self.current_weights = None
        self.current_sum = None

    def get_clipped_model(self):
        """
        Computes parameter values with clipping-based defense and returns a model state dictionary.
        
        This method:
        1. Analyzes the distribution of received parameter values
        2. Computes reasonable clipping bounds based on statistical measures
        3. Clips outlier values to prevent malicious influence
        4. Computes the mean of clipped values
        """
        # Create a tensor to hold the clipped and averaged values
        clipped_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
        
        clipped_count = 0
        total_processed = 0
        
        # Process each parameter position
        for idx in range(self.total_length):
            if idx in self.param_values and len(self.param_values[idx]) > 0:
                # Get values for this parameter position
                values = self.param_values[idx]
                filtered_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                
                if len(filtered_values) > 0:
                    total_processed += len(filtered_values)
                    
                    # Compute clipping bounds
                    lower_bound, upper_bound = self._get_adaptive_clipping_bounds(idx, filtered_values)
                    
                    # Store bounds for analysis
                    self.clipping_bounds_history[idx] = (lower_bound, upper_bound)
                    
                    # Apply clipping
                    clipped_values = []
                    for val in filtered_values:
                        if val < lower_bound:
                            clipped_values.append(lower_bound)
                            clipped_count += 1
                        elif val > upper_bound:
                            clipped_values.append(upper_bound)
                            clipped_count += 1
                        else:
                            clipped_values.append(val)
                    
                    # Compute mean of clipped values
                    mean_val = np.mean(clipped_values)
                    
                    # Update parameter history for adaptive clipping
                    self._update_parameter_history(idx, mean_val)
                    
                    # Ensure the result is valid
                    if np.isnan(mean_val) or np.isinf(mean_val):
                        mean_val = 0.0
                    
                    clipped_weights[idx] = torch.tensor(
                        mean_val,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    clipped_weights[idx] = 0.0
            else:
                clipped_weights[idx] = 0.0
        
        # Update clipping statistics
        self.clipped_weights_count = clipped_count
        self.total_weights_processed = total_processed
        
        # Final validation of clipped weights
        if torch.any(torch.isnan(clipped_weights)) or torch.any(torch.isinf(clipped_weights)):
            clipped_weights = self._detect_and_sanitize_nan_inf(
                clipped_weights, 
                "clipped_weights", 
                f"node_{self.uid}"
            )
        
        # Convert flat tensor back to model state dict
        state_dict = self._post_step(clipped_weights)
        
        # Final validation of the state dict
        state_dict, was_corrupted = self._validate_model_state(state_dict, f"clipped_model_node_{self.uid}")
        
        return state_dict

    def defender_finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging for defender nodes.
        Applies clipping-based defense against poisoning attacks.
        """
        # Process all incoming data
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        # Compute adversarial proportion statistics
        max_adv_prop, adv_stats = self.compute_max_adversarial_proportion()
        self.max_adversarial_proportion = max_adv_prop

        # Calculate the clipped parameter values and update model
        clipped_model = self.get_clipped_model()
        self.model.load_state_dict(clipped_model)
    
        # Save clipping metrics to file
        self._save_clipping_metrics()
    
        # Clean up for the next round
        self.communication_round += 1
        self.param_values = None
        self.param_adversarial_sources = None  # Reset adversarial tracking
        # Reset corruption counters for next round
        self.nan_inf_detection_count = 0
        self.corrupted_weights_received = 0
        self.corrupted_weights_rejected = 0

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.
        """
        if self.uid in self.adversarial_nodes:
            self.adversarial_finish_forward_averaging(peer_deques)
        else:
            self.defender_finish_forward_averaging(peer_deques)
    
    def get_corruption_diagnostics(self):
        """
        Get comprehensive corruption diagnostics for this node.
        """
        return {
            "node_id": self.uid,
            "node_type": "adversarial" if self.uid in self.adversarial_nodes else "defender",
            "attack_type": getattr(self, 'attack_type', 'unknown'),
            "communication_round": getattr(self, 'communication_round', 0),
            "corruption_stats": {
                "total_corruption_events": self.nan_inf_detection_count,
                "total_corrupted_weights_received": self.corrupted_weights_received,
                "total_corrupted_weights_sanitized": self.corrupted_weights_rejected,
                "corruption_rate": self.corrupted_weights_received / max(1, self.total_weights_count) if hasattr(self, 'total_weights_count') else 0.0
            },
            "clipping_stats": {
                "clipped_weights_count": self.clipped_weights_count,
                "total_weights_processed": self.total_weights_processed,
                "clipping_rate": self.clipped_weights_count / max(1, self.total_weights_processed),
                "clipping_factor": self.clipping_factor,
                "percentile_threshold": self.percentile_threshold,
                "adaptive_clipping": self.adaptive_clipping
            },
            "adversarial_influence": {
                "max_adversarial_proportion": getattr(self, 'max_adversarial_proportion', 0.0),
                "adversarial_nodes_in_network": self.adversarial_nodes
            }
        }

    def _save_clipping_metrics(self):
        """
        Save clipping metrics and adversarial proportion data.
        """
        try:
            # Compute adversarial proportion statistics
            max_adv_prop, adv_stats = self.compute_max_adversarial_proportion()
            self.max_adversarial_proportion = max_adv_prop
            
            metrics_file = os.path.join(self.log_dir, f"clipping_metrics_{self.uid}.json")
            
            # Prepare metrics data
            metrics_data = {
                "round": self.communication_round,
                "max_adversarial_proportion": max_adv_prop,
                "clipping_stats": {
                    "clipped_weights": self.clipped_weights_count,
                    "total_weights": self.total_weights_processed,
                    "clipping_rate": self.clipped_weights_count / max(1, self.total_weights_processed),
                    "clipping_factor": self.clipping_factor,
                    "percentile_threshold": self.percentile_threshold
                }
            }
            
            # Load existing data if file exists
            existing_data = {}
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_data = {}
            
            # Update with new data
            existing_data[str(self.communication_round)] = metrics_data
            
            # Save back to file
            with open(metrics_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            pass
    
    def compute_max_adversarial_proportion(self):
        """
        Compute the maximum proportion of adversarial values across all weight positions.
        Returns the maximum proportion and detailed statistics.
        """
        if self.param_adversarial_sources is None or len(self.param_adversarial_sources) == 0:
            return 0.0, {}
        
        adversarial_proportions = []
        weight_stats = {}
        
        for idx in range(self.total_length):
            if idx in self.param_adversarial_sources and len(self.param_adversarial_sources[idx]) > 0:
                sources = self.param_adversarial_sources[idx]
                total_values = len(sources)
                adversarial_count = sum(1 for _, is_adversarial in sources if is_adversarial)
                
                if total_values > 0:
                    proportion = adversarial_count / total_values
                    adversarial_proportions.append(proportion)
                    
                    # Store detailed stats for this weight
                    weight_stats[idx] = {
                        'total_values': total_values,
                        'adversarial_count': adversarial_count,
                        'proportion': proportion,
                        'adversarial_sources': [node for node, is_adv in sources if is_adv]
                    }
        
        if adversarial_proportions:
            max_proportion = max(adversarial_proportions)
            avg_proportion = sum(adversarial_proportions) / len(adversarial_proportions)
            
            # Find the weight(s) with maximum adversarial proportion
            max_weights = [idx for idx, stats in weight_stats.items() 
                          if stats['proportion'] == max_proportion]
            
            detailed_stats = {
                'max_proportion': max_proportion,
                'avg_proportion': avg_proportion,
                'weights_with_max_proportion': max_weights,
                'total_weights_tracked': len(adversarial_proportions),
                'weight_details': weight_stats
            }
            
            return max_proportion, detailed_stats
        else:
            return 0.0, {}
    
    def get_adversarial_proportion_stats(self):
        """
        Get detailed statistics about adversarial proportions for external analysis.
        """
        if self.param_adversarial_sources is None:
            return {
                "max_proportion": 0.0,
                "status": "no_data_available"
            }
        
        max_proportion, detailed_stats = self.compute_max_adversarial_proportion()
        
        # Add summary information
        result = {
            "node_id": self.uid,
            "round": self.communication_round,
            "max_adversarial_proportion": max_proportion,
            "detailed_stats": detailed_stats,
            "summary": {
                "total_weights": self.total_length,
                "weights_with_adversarial_input": len(detailed_stats.get('weight_details', {})),
                "adversarial_nodes_in_network": self.adversarial_nodes,
                "is_defender_node": self.uid not in self.adversarial_nodes
            }
        }
        
        return result

    def get_clipping_bounds_summary(self):
        """
        Get a summary of current clipping bounds for analysis.
        """
        if not self.clipping_bounds_history:
            return {}
        
        bounds_summary = {}
        for param_idx, (lower, upper) in self.clipping_bounds_history.items():
            bounds_summary[param_idx] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "range": upper - lower
            }
        
        # Compute overall statistics
        ranges = [bounds["range"] for bounds in bounds_summary.values()]
        overall_stats = {
            "mean_range": np.mean(ranges) if ranges else 0.0,
            "median_range": np.median(ranges) if ranges else 0.0,
            "min_range": np.min(ranges) if ranges else 0.0,
            "max_range": np.max(ranges) if ranges else 0.0,
            "total_parameters": len(bounds_summary)
        }
        
        return {
            "individual_bounds": bounds_summary,
            "overall_stats": overall_stats
        }
