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
    Poisoned model sharing class that sends malicious gradients
    Implements various poisoning strategies for adversarial attacks
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
        # log_poisoning_metrics=True,
    ):
        """
        Constructor for poisoning class
        
        Parameters
        ----------
        rank, machine_id, etc. : same as parent class
        attack_type : str
            Poisoning strategy ('zero', 'flip', 'noise', 'scale')
        adversarial_nodes : list
            List of node IDs that will perform the attack
        poison_period : int
            Number of rounds between poisonings
        log_poisoning_metrics : bool
            Whether to log poisoning metrics
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

        self.param_values = None
        
        # Add diagnostic tracking for NaN/Inf issues
        self.nan_inf_detection_count = 0
        self.corrupted_weights_received = 0
        self.corrupted_weights_rejected = 0

    def _detect_and_sanitize_nan_inf(self, tensor, tensor_name="tensor", sender_node="unknown", default_value=0.0):
        """
        Detect and sanitize NaN/Inf values in tensors.
        Returns sanitized tensor and logs detailed diagnostics.
        """
        if tensor is None:
            logging.error(f"Node {self.uid}: Received None tensor from {sender_node} for {tensor_name}")
            return torch.zeros_like(tensor) if tensor is not None else None
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
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
            
            # Log detailed diagnostics
            logging.warning(f"Node {self.uid}: CORRUPTION DETECTED in {tensor_name} from {sender_node}:")
            logging.warning(f"  - Tensor shape: {original_shape}, dtype: {original_dtype}")
            logging.warning(f"  - NaN count: {nan_count}")
            logging.warning(f"  - Inf count: {inf_count}")
            logging.warning(f"  - Large values (>1e10): {large_count}")
            logging.warning(f"  - Total corrupted: {total_corrupted}/{tensor.numel()}")
            
            if tensor.numel() > 0:
                # Get statistics of non-corrupted values
                valid_mask = ~(nan_mask | inf_mask)
                if torch.any(valid_mask):
                    valid_values = tensor[valid_mask]
                    logging.warning(f"  - Valid values stats: min={valid_values.min():.6f}, max={valid_values.max():.6f}, mean={valid_values.mean():.6f}, std={valid_values.std():.6f}")
                else:
                    logging.warning(f"  - NO VALID VALUES FOUND - all values are NaN/Inf!")
            
            # Sanitize the tensor
            sanitized_tensor = tensor.clone()
            
            # Replace NaN with default value
            if nan_count > 0:
                sanitized_tensor[nan_mask] = default_value
                logging.warning(f"  - Replaced {nan_count} NaN values with {default_value}")
            
            # Replace Inf with clamped values
            if inf_count > 0:
                # Replace positive inf with large positive value, negative inf with large negative value
                pos_inf_mask = torch.isposinf(tensor)
                neg_inf_mask = torch.isneginf(tensor)
                sanitized_tensor[pos_inf_mask] = 1e6  # Large but finite positive value
                sanitized_tensor[neg_inf_mask] = -1e6  # Large but finite negative value
                logging.warning(f"  - Replaced {torch.sum(pos_inf_mask).item()} +Inf and {torch.sum(neg_inf_mask).item()} -Inf values")
            
            # Optionally clamp extremely large values
            if large_count > 0:
                sanitized_tensor = torch.clamp(sanitized_tensor, min=-1e8, max=1e8)
                logging.warning(f"  - Clamped {large_count} extremely large values to [-1e8, 1e8] range")
            
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
                
                logging.error(f"Node {self.uid}: Corrupted layer '{layer_name}' from {source}: {nan_count} NaNs, {inf_count} Infs")
                
                # Sanitize this layer
                sanitized_tensor = self._detect_and_sanitize_nan_inf(tensor, f"layer_{layer_name}", source)
                state_dict[layer_name] = sanitized_tensor
        
        if corrupted_layers:
            logging.error(f"Node {self.uid}: Total corruption in model from {source}: {len(corrupted_layers)} layers, {total_corrupted} corrupted values")
            logging.error(f"Node {self.uid}: Corrupted layers: {corrupted_layers}")
        
        return state_dict, len(corrupted_layers) > 0

    
    def _apply_model_poisoning(self):
        """
        Apply poisoning strategy for adversarial nodes
        """
        # Check if we should poison in this round (based on poison_after)
        if self.poison_after > 1 and self.communication_round % self.poison_after != 0:
            logging.debug(f"Node {self.uid} skipping poisoning in round {self.communication_round}")
            # Just apply the averaged model when poisoning is skipped
            mean_state_dict = self._post_step(self.current_sum)
            # Validate and sanitize before applying
            mean_state_dict, was_corrupted = self._validate_model_state(mean_state_dict, f"mean_averaging_node_{self.uid}")
            if was_corrupted:
                logging.warning(f"Node {self.uid}: Detected and fixed corruption in mean state during skip-poisoning")
            self.model.load_state_dict(mean_state_dict)
            return
        
        logging.info(f"Node {self.uid} applying {self.attack_type} poisoning in round {self.communication_round}")
    
        if self.attack_type == 'zero':
            # Zero out all parameters
            self.current_sum.zero_()
            zero_state_dict = self._post_step(self.current_sum)
            # Validate the zero state (should be clean but let's check)
            zero_state_dict, was_corrupted = self._validate_model_state(zero_state_dict, f"zero_attack_node_{self.uid}")
            if was_corrupted:
                logging.error(f"Node {self.uid}: Unexpected corruption in zero attack state!")
            self.model.load_state_dict(zero_state_dict)
        
        elif self.attack_type == 'flip_grad':
            # Mirror the gradient update to move away from consensus
            mean_state_dict = self._post_step(self.current_sum)
            init_state_dict = copy.deepcopy(self.model.state_dict())
            
            # First validate input states
            mean_state_dict, mean_corrupted = self._validate_model_state(mean_state_dict, f"mean_state_node_{self.uid}")
            init_state_dict, init_corrupted = self._validate_model_state(init_state_dict, f"init_state_node_{self.uid}")
            
            if mean_corrupted:
                logging.warning(f"Node {self.uid}: Fixed corruption in mean state before flip_grad")
            if init_corrupted:
                logging.warning(f"Node {self.uid}: Fixed corruption in init state before flip_grad")
            
            final_state_dict = {}
            corrupted_keys = []
            
            for key in init_state_dict:
                # Perform flip operation: new = 2 * init - mean
                try:
                    flipped_update = 2 * init_state_dict[key] - mean_state_dict[key]
                    
                    # Check for NaN/Inf in the result
                    if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                        corrupted_keys.append(key)
                        logging.warning(f"Node {self.uid}: NaN/Inf detected in {key} during flip_grad calculation")
                        
                        # Try alternative: use sanitized version
                        flipped_update = self._detect_and_sanitize_nan_inf(
                            flipped_update, 
                            f"flip_grad_{key}", 
                            f"node_{self.uid}",
                            default_value=0.0
                        )
                        
                        # If still problematic, fallback to mean state
                        if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                            logging.error(f"Node {self.uid}: Failed to sanitize {key}, using mean state as fallback")
                            final_state_dict[key] = mean_state_dict[key]
                        else:
                            final_state_dict[key] = flipped_update
                    else:
                        # Check for extremely large values that might become problematic
                        if torch.max(torch.abs(flipped_update)) > 1e8:
                            logging.warning(f"Node {self.uid}: Very large values detected in {key} during flip_grad, clamping")
                            flipped_update = torch.clamp(flipped_update, min=-1e8, max=1e8)
                        final_state_dict[key] = flipped_update
                        
                except Exception as e:
                    logging.error(f"Node {self.uid}: Exception during flip_grad for {key}: {e}")
                    final_state_dict[key] = mean_state_dict[key]
            
            if corrupted_keys:
                logging.warning(f"Node {self.uid}: Handled corruption in {len(corrupted_keys)} layers during flip_grad: {corrupted_keys}")
            
            # Final validation before loading
            final_state_dict, final_corrupted = self._validate_model_state(final_state_dict, f"final_flip_grad_node_{self.uid}")
            if final_corrupted:
                logging.error(f"Node {self.uid}: Still found corruption after flip_grad sanitization!")
            
            self.model.load_state_dict(final_state_dict)
        
        else:
            logging.warning(f"Unknown attack type: {self.attack_type}")
            # Use normal averaged model for unknown attack types
            mean_state_dict = self._post_step(self.current_sum)
            mean_state_dict, was_corrupted = self._validate_model_state(mean_state_dict, f"fallback_node_{self.uid}")
            if was_corrupted:
                logging.warning(f"Node {self.uid}: Fixed corruption in fallback mean state")
            self.model.load_state_dict(mean_state_dict)

    def _apply_defense(self):
        """
        Apply defense mechanisms against poisoning attacks

        This function should be implemented with defense strategies
        to mitigate the effects of poisoning attacks
        """
        # TODO: Implement defense mechanisms here
        pass

    def forward_averaging(self, data):
        if self.uid in self.adversarial_nodes:
            self.adversarial_forward_averaging(data)
        else:
            self.defender_forward_averaging(data)

    def adversarial_forward_averaging(self, data):
        """
        Computes the sum for the average in a state based manner.

        Parameters
        ----------
        data : dict
            Received data

        Returns
        -------
        None

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
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        logging.debug(
            "Forward Averaging model from neighbor {} of iteration {}".format(
                sender_node, iteration
            )
        )
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        
        logging.debug("Deserialized model from neighbor {}".format(sender_node))
        
        # Validate received weights for NaN/Inf before adding them
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            logging.warning(f"Node {self.uid}: Corrupted weights received from {sender_node}, sanitizing...")
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )
        
        # Also validate current_sum before updating
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            logging.error(f"Node {self.uid}: Corruption detected in current_sum before update, sanitizing...")
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_before_update", 
                f"node_{self.uid}"
            )
        
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1
        
        # Validate current_sum after update
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            logging.error(f"Node {self.uid}: Corruption detected in current_sum after update from {sender_node}, sanitizing...")
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_update", 
                f"node_{self.uid}"
            )

    def defender_forward_averaging(self, data):
        """
        Computes and collects parameter values for median-based aggregation.
        param_values is a dictionary with the weight index as key and the list of received weights as value.
        """
        if self.param_values is None:
            # First time, initialize param_values dictionary
            self.param_values = defaultdict(list)
        
            # Add own model weights as the first entry
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            own_weights = torch.cat(tensors_to_cat, dim=0).to(self.device)
        
            # Add own weights to param_values
            for i in range(len(own_weights)):
                self.param_values[i].append(own_weights[i].item())

        # Process received model data
        iteration = data["iteration"]
        sender_node = data.get("real_node", data.get("vSource", "unknown"))  # Get sender info
        
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        
        # Remove real_node from data if it exists (after extracting it)
        if "real_node" in data:
            del data["real_node"]
            
        logging.debug(
            "Forward Averaging model from neighbor {} of iteration {}".format(
                sender_node, iteration
            )
        )
    
        # Deserialize received model data
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
    
        logging.debug("Deserialized model from neighbor {}".format(sender_node))
        
        # Validate received weights for NaN/Inf before processing
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            logging.warning(f"Node {self.uid}: Corrupted weights received from {sender_node}, sanitizing...")
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )
    
        # Add the received weights to the param_values dictionary
        for idx, value in zip(indices.tolist(), deserializedT.tolist()):
            # Additional check for individual values
            if np.isnan(value) or np.isinf(value):
                logging.warning(f"Node {self.uid}: Skipping corrupted value {value} at index {idx} from {sender_node}")
                value = 0.0  # Replace with safe default
            self.param_values[idx].append(value)
        
    
    def adversarial_finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging for adversarial nodes.

        This method is called after all peer deques have been processed.
        It applies the model poisoning strategy if the node is adversarial.

        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        # Validate current_sum and current_weights before final processing
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            logging.error(f"Node {self.uid}: Corruption in current_sum before final processing, sanitizing...")
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_final", 
                f"node_{self.uid}"
            )
        
        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            logging.error(f"Node {self.uid}: Corruption in current_weights before final processing, sanitizing...")
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
            logging.warning(f"Node {self.uid}: Found {torch.sum(zero_weights_mask).item()} zero weights, replacing with 1.0")
            self.current_weights[zero_weights_mask] = 1.0
        
        self.current_weights = 1.0 / self.current_weights
        
        # Check for NaN/Inf after division
        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            logging.error(f"Node {self.uid}: Corruption in current_weights after division, sanitizing...")
            self.current_weights = self._detect_and_sanitize_nan_inf(
                self.current_weights, 
                "current_weights_after_division", 
                f"node_{self.uid}",
                default_value=1.0
            )
        
        self.current_sum = self.current_sum * self.current_weights
        
        # Final validation of current_sum
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            logging.error(f"Node {self.uid}: Corruption in current_sum after multiplication, sanitizing...")
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_multiplication", 
                f"node_{self.uid}"
            )
        
        logging.debug("Finished averaging")
        self.current_sum = self.current_sum.cpu()

        self._apply_model_poisoning()
        
        # Log corruption summary for adversarial nodes if there was corruption
        if self.nan_inf_detection_count > 0:
            logging.warning(f"Adversarial Node {self.uid} Round {self.communication_round}: "
                          f"Detected {self.nan_inf_detection_count} corruption events during poisoning")
            if self.communication_round % 10 == 0 or self.nan_inf_detection_count > 5:
                self.log_corruption_summary()
        
        self.communication_round += 1
        self.current_weights = None
        self.current_sum = None

    def get_medgrad_model(self):
        """
        Computes the element-wise median of collected weight values
        and returns a model state dictionary with these medians.
        """
        # Create a tensor to hold the median values
        median_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
    
        # Calculate the median for each weight position
        corrupted_indices = []
        for idx in range(self.total_length):
            if idx in self.param_values and len(self.param_values[idx]) > 0:
                # Filter out any remaining NaN/Inf values before computing median
                values = self.param_values[idx]
                filtered_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                
                if len(filtered_values) > 0:
                    median_val = np.median(filtered_values)
                    if np.isnan(median_val) or np.isinf(median_val):
                        logging.warning(f"Node {self.uid}: Computed median is corrupted at index {idx}, using 0.0")
                        median_val = 0.0
                        corrupted_indices.append(idx)
                    median_weights[idx] = torch.tensor(
                        median_val,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    logging.warning(f"Node {self.uid}: No valid values for index {idx}, using 0.0")
                    corrupted_indices.append(idx)
                    median_weights[idx] = 0.0
        
        if corrupted_indices:
            logging.warning(f"Node {self.uid}: Found {len(corrupted_indices)} corrupted indices in median computation")
    
        # Final validation of median weights
        if torch.any(torch.isnan(median_weights)) or torch.any(torch.isinf(median_weights)):
            logging.error(f"Node {self.uid}: Corruption in median_weights, sanitizing...")
            median_weights = self._detect_and_sanitize_nan_inf(
                median_weights, 
                "median_weights", 
                f"node_{self.uid}"
            )
    
        # Convert flat tensor back to model state dict
        state_dict = self._post_step(median_weights)
        
        # Final validation of the state dict
        state_dict, was_corrupted = self._validate_model_state(state_dict, f"median_model_node_{self.uid}")
        if was_corrupted:
            logging.error(f"Node {self.uid}: Fixed corruption in final median model state")
        
        return state_dict

    def defender_finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging for defender nodes.
        Applies median-based defense against poisoning attacks.
        """
        # Process all incoming data
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        # Log corruption statistics
        if self.nan_inf_detection_count > 0:
            logging.warning(f"Node {self.uid} Round {self.communication_round}: "
                          f"Detected {self.nan_inf_detection_count} corruption events, "
                          f"received {self.corrupted_weights_received} corrupted weights, "
                          f"rejected/sanitized {self.corrupted_weights_rejected} values")
            # Log detailed corruption summary every 10 rounds or when there's significant corruption
            if self.communication_round % 10 == 0 or self.nan_inf_detection_count > 5:
                self.log_corruption_summary()

        # Calculate the weight-wise median and update model
        medgrad_model = self.get_medgrad_model()
        self.model.load_state_dict(medgrad_model)
    
        logging.debug("Finished median-based averaging")
    
        # Save corruption metrics to file
        self._save_corruption_metrics()
    
        # Clean up for the next round
        self.communication_round += 1
        self.param_values = None
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
            }
        }

    def log_corruption_summary(self):
        """
        Log a comprehensive summary of corruption events for diagnostic purposes.
        """
        diagnostics = self.get_corruption_diagnostics()
        logging.info(f"=== CORRUPTION SUMMARY for Node {self.uid} ===")
        logging.info(f"  Node Type: {diagnostics['node_type']}")
        logging.info(f"  Attack Type: {diagnostics['attack_type']}")
        logging.info(f"  Round: {diagnostics['communication_round']}")
        logging.info(f"  Total Corruption Events: {diagnostics['corruption_stats']['total_corruption_events']}")
        logging.info(f"  Corrupted Weights Received: {diagnostics['corruption_stats']['total_corrupted_weights_received']}")
        logging.info(f"  Corrupted Weights Sanitized: {diagnostics['corruption_stats']['total_corrupted_weights_sanitized']}")
        logging.info(f"  Corruption Rate: {diagnostics['corruption_stats']['corruption_rate']:.4f}")
        logging.info("=== END CORRUPTION SUMMARY ===")

    def _save_corruption_metrics(self):
        """
        Save corruption metrics to a JSON file for analysis.
        """
        try:
            metrics = {
                "node_id": self.uid,
                "current_round": self.communication_round,
                "corruption_stats": {
                    "nan_inf_detection_count": self.nan_inf_detection_count,
                    "corrupted_weights_received": self.corrupted_weights_received,
                    "corrupted_weights_rejected": self.corrupted_weights_rejected
                }
            }
            
            metrics_file = os.path.join(self.log_dir, f"corruption_metrics_{self.uid}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save corruption metrics for node {self.uid}: {e}")







