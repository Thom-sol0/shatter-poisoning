import torch
import numpy as np
from collections import defaultdict
import os
import json
from virtualNodes.sharing.VNodeSharingDefenseWeightTracker import VNodeSharingDefenseWeightTracker as VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    Cone Clipping defense implementation.
    
    This class implements a direction-based clipping defense mechanism where each honest node
    limits not just the magnitude but also the direction of incoming updates. Updates with
    cosine similarity below a threshold are rotated to be more aligned with a characteristic
    direction (median of previous updates). This defends against attackers trying to shift the
    model in abnormal directions.
    
    The cosine similarity threshold is dynamically adjusted based on the median of observed
    similarities, similar to how the norm threshold works in NormClippingDynamic.
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
        save_interval=10,
        experiment_id=None,
        tau_own=1.0,  # Norm threshold for own model update 
        tau_init=0.1,  # Initial norm threshold for neighbor updates
        cos_init=0.1,  # Initial cosine similarity threshold
        update_window=5,  # Number of rounds between updating thresholds
    ):
        """
        Constructor for cone clipping defense class.
        
        Parameters:
        -----------
        rank, machine_id, etc. : same as parent class
        tau_own : float
            Norm threshold for the node's own model update
        tau_init : float 
            Initial norm threshold for neighbor updates
        cos_init : float
            Initial cosine similarity threshold (-1 to 1, higher is more strict)
        update_window : int
            Number of rounds between updating thresholds
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
            save_interval=save_interval,
            experiment_id=experiment_id
        )
        
        self.tau_own = tau_own
        self.tau_nbr = tau_init  # Current norm threshold
        self.cos_threshold = cos_init  # Current cosine threshold
        self.cos_init = cos_init  # Store initial value for reference
        self.update_window = update_window
        
        # For tracking direction history
        self.update_history = []  # Stores full update vectors for computing characteristic direction
        self.last_threshold_update = 0
        
        # The characteristic direction (starts as None until we have enough data)
        self.characteristic_direction = None
        
        # For tracking cosine similarities
        self.cosine_similarities = []
        self.cosine_history = []
        
        # For logging threshold changes
        self.direction_history = [(0, None)]
        self.cos_threshold_history = [(0, cos_init)]

    def initialize_defense_data(self):
        """Initialize data structures for cone clipping defense"""
        self.defense_data = {
            'neighbor_weights': defaultdict(list),  # Map node -> weight tensor
            'param_adversarial_sources': defaultdict(list),  # Track adversarial sources
            'round_updates': [],  # Track updates observed in current round
            'round_cos_similarities': []  # Track cosine similarities in current round
        }
    
    def _compute_characteristic_direction(self):
        """
        Compute the characteristic direction based on accumulated updates.
        This represents the "normal" or "expected" direction of model updates.
        """
        if not self.update_history:
            return None
            
        # Stack all updates and compute the median along the first dimension
        stacked_updates = torch.stack(self.update_history)
        
        # Normalize each update vector to focus on direction, not magnitude
        update_norms = torch.norm(stacked_updates, p=2, dim=1, keepdim=True)
        update_norms = torch.clamp(update_norms, min=1e-8)  # Avoid division by zero
        normalized_updates = stacked_updates / update_norms
        
        # Compute element-wise median for characteristic direction
        characteristic_dir = torch.median(normalized_updates, dim=0).values
        
        # Normalize the characteristic direction
        char_dir_norm = torch.norm(characteristic_dir, p=2)
        if char_dir_norm > 1e-8:  # Avoid division by zero
            characteristic_dir = characteristic_dir / char_dir_norm
            
        return characteristic_dir
    
    def _update_thresholds(self):
        """
        Update both the characteristic direction and the cosine similarity threshold
        """
        # First update the characteristic direction
        if len(self.update_history) > 0:
            # Compute new characteristic direction
            new_direction = self._compute_characteristic_direction()
            
            if new_direction is not None:
                # Update the characteristic direction
                self.characteristic_direction = new_direction
                
                # Log direction change (store only the norm since the full vector is too large)
                self.direction_history.append((self.communication_round, torch.norm(new_direction).item()))
                
                # Log update to console
                print(f"Node {self.uid}: Updated characteristic direction at round {self.communication_round}")
        
        # Now update the cosine similarity threshold if we have enough data
        if len(self.cosine_similarities) > 0:
            # Compute median cosine similarity
            median_cosine = np.median(self.cosine_similarities)

            print(f"Node {self.uid}: Median cosine similarity for round {self.communication_round} is {median_cosine:.4f}")
            
            # Update threshold, but ensure it doesn't go below a minimum (original init value)
            # This prevents the threshold from becoming too relaxed if many adversarial updates exist
            old_threshold = self.cos_threshold
            self.cos_threshold = 1.5 * max(median_cosine, self.cos_init)
            
            # Log threshold change
            self.cos_threshold_history.append((self.communication_round, self.cos_threshold))
            
            # Log update to console
            print(f"Node {self.uid}: Updated cosine threshold from {old_threshold:.4f} to {self.cos_threshold:.4f}")
        
        # Manage update history size
        max_history = self.update_window  # Keep at most this many updates
        if len(self.update_history) > max_history:
            self.update_history = self.update_history[-max_history:]
        
        self.last_threshold_update = self.communication_round
        
        # Store current cosine similarities in history and reset
        self.cosine_history.extend(self.cosine_similarities)
        self.cosine_similarities = []
        
    def defender_forward_averaging(self, data):
        """
        Process incoming updates and apply both norm clipping and directional clipping
        to ensure updates remain within expected norms and directions.
        """
        # Initialize defense data if not done
        if self.defense_data is None:
            self.initialize_defense_data()
            
            # Store own model weights first
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            own_weights = torch.cat(tensors_to_cat, dim=0).to(self.device)
            self.defense_data['neighbor_weights'][self.uid] = own_weights
            
        # Process received data
        sender_node = data.get("real_node", data.get("vSource", "unknown"))

        self.neighbor_list.append(sender_node)

        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print(f"uid: {self.uid} | Exception: {e}")
            raise e
            
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT,
                f"received_weights_from_{sender_node}",
                sender_node
            )
            
        # Initialize neighbor's weights if first time seeing them
        if sender_node not in self.defense_data['neighbor_weights']:
            self.defense_data['neighbor_weights'][sender_node] = self.defense_data['neighbor_weights'][self.uid].clone()

        # Apply clipping to weight differences (updates) from neighbors
        if sender_node != self.uid:
            # Get the corresponding chunk of our own weights using indices
            own_chunk = torch.index_select(self.defense_data['neighbor_weights'][self.uid], 0, indices)
            
            # Compute weight difference (update) from our current weights
            weight_diff = deserializedT - own_chunk
            
            # First apply standard norm clipping
            norm = torch.norm(weight_diff, p=2).item()
            
            if norm > self.tau_nbr:
                # Scale down the difference to have norm = tau_nbr
                scale_factor = self.tau_nbr / norm
                weight_diff = weight_diff * scale_factor
                norm = self.tau_nbr  # Update norm value
            
            # Then apply directional (cosine similarity) clipping if we have a characteristic direction
            if self.characteristic_direction is not None:
                # Get the portion of the characteristic direction corresponding to this chunk
                char_dir_chunk = torch.index_select(self.characteristic_direction, 0, indices)
                
                # Compute cosine similarity
                weight_diff_norm = torch.norm(weight_diff, p=2)
                char_dir_chunk_norm = torch.norm(char_dir_chunk, p=2)
                
                # Avoid division by zero
                if weight_diff_norm > 1e-8 and char_dir_chunk_norm > 1e-8:
                    cosine_sim = torch.sum(weight_diff * char_dir_chunk) / (weight_diff_norm * char_dir_chunk_norm)
                    cosine_sim = cosine_sim.item()  # Convert to Python scalar
                    
                    # Track cosine similarity
                    self.cosine_similarities.append(cosine_sim)
                    self.defense_data['round_cos_similarities'].append(cosine_sim)

                    print(f"current cosine threshold: {self.cos_threshold:.4f}")

                    # If cosine similarity is too low (update direction is too different from characteristic direction)
                    if cosine_sim > self.cos_threshold:
                        # Decompose weight_diff into components parallel and perpendicular to char_dir_chunk
                        parallel_component = (torch.sum(weight_diff * char_dir_chunk) / char_dir_chunk_norm**2) * char_dir_chunk
                        perpendicular_component = weight_diff - parallel_component
                        
                        # Compute the required scaling to achieve target cosine similarity
                        perp_norm = torch.norm(perpendicular_component, p=2)
                        para_norm = torch.norm(parallel_component, p=2)
                        
                        if perp_norm > 1e-8 and para_norm > 1e-8:
                            # Target perpendicular component to achieve desired cosine similarity
                            # cos_sim = para_norm / sqrt(para_norm^2 + perp_norm^2)
                            # Solve for new_perp_norm: cos_threshold = para_norm / sqrt(para_norm^2 + new_perp_norm^2)
                            target_perp_norm = para_norm * np.sqrt((1 - self.cos_threshold**2) / max(self.cos_threshold**2, 1e-8))
                            
                            # Scale perpendicular component
                            scale_factor = min(target_perp_norm / perp_norm, 1.0)  # Never increase perpendicular component
                            perpendicular_component = perpendicular_component * scale_factor
                            
                            # Reconstruct weight_diff with adjusted perpendicular component
                            weight_diff = parallel_component + perpendicular_component
                            
                            print(f"Node {self.uid} adjusted update from {sender_node}: cosine sim {cosine_sim:.4f} -> {self.cos_threshold:.4f}")
            
            # Store the update for computing characteristic direction
            if len(indices) == self.total_length:  # Only store full updates
                self.update_history.append(weight_diff.detach().clone())
            else:
                # For partial updates, we can expand to full size
                full_update = torch.zeros(self.total_length, device=self.device)
                full_update[indices] = weight_diff
                self.update_history.append(full_update.detach().clone())
                
            # Apply the (potentially modified) update
            deserializedT = own_chunk + weight_diff
        
        # Store the clipped chunk using scatter_
        self.defense_data['neighbor_weights'][sender_node].scatter_(0, indices, deserializedT)
        
        # Track adversarial sources 
        sender_is_adversarial = sender_node in self.adversarial_nodes
        self.defense_data['param_adversarial_sources'][sender_node] = sender_is_adversarial

    def get_defended_model(self):
        """
        Return the averaged model after applying cone clipping defense.
        Also updates thresholds if needed.
        """
        # Check if it's time to update thresholds
        rounds_since_update = self.communication_round - self.last_threshold_update
        if (rounds_since_update >= self.update_window and self.communication_round > 0) or self.characteristic_direction is None:
            self._update_thresholds()
            
        # Get all neighbor updates after clipping
        neighbor_updates = list(self.defense_data['neighbor_weights'].values())
        
        # Simple average of clipped updates
        defended_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
        for weights in neighbor_updates:
            defended_weights += weights
        defended_weights = defended_weights / len(neighbor_updates)
        
        # Sanitize any remaining NaN/Inf values
        if torch.any(torch.isnan(defended_weights)) or torch.any(torch.isinf(defended_weights)):
            defended_weights = self._detect_and_sanitize_nan_inf(
                defended_weights,
                "defended_weights",
                f"node_{self.uid}"
            )
            
        # Convert back to state dict
        state_dict = self._post_step(defended_weights)
        state_dict, was_corrupted = self._validate_model_state(state_dict, f"defended_model_node_{self.uid}")
        
        # Save direction and cosine data
        self._save_direction_history()
        
        return state_dict

    def _save_direction_history(self):
        """
        Save direction history, threshold changes, and cosine similarity statistics
        """
        try:
            import json
            import os
            
            history_file = os.path.join(self.log_dir, f"cone_clipping_{self.uid}.json")
            
            # Calculate statistics on cosine similarities
            cos_stats = {}
            if self.cosine_similarities:
                cos_stats = {
                    'min': min(self.cosine_similarities),
                    'max': max(self.cosine_similarities),
                    'mean': sum(self.cosine_similarities) / len(self.cosine_similarities),
                    'median': sorted(self.cosine_similarities)[len(self.cosine_similarities)//2],
                    'count': len(self.cosine_similarities)
                }
            
            with open(history_file, 'w') as f:
                json.dump({
                    'direction_history': self.direction_history,
                    'cosine_threshold_history': self.cos_threshold_history,
                    'current_cosine_threshold': self.cos_threshold,
                    'initial_cosine_threshold': self.cos_init,
                    'update_window': self.update_window,
                    'norm_threshold': self.tau_nbr,
                    'cosine_stats': cos_stats,
                    'communication_round': self.communication_round
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving direction history: {e}")

    def _cleanup_defense_data(self):
        """Clean up defense data structures between rounds"""
        # Keep the update history for characteristic direction calculation
        # but clear the other defense data
        self.defense_data = None