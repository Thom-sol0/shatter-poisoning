import torch
import numpy as np
from collections import defaultdict
from virtualNodes.sharing.VNodeSharingDefenseBase import VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    Chunk-specific Dynamic Norm Clipping defense implementation.
    
    This class implements an adaptive norm clipping defense mechanism where the clipping
    threshold is dynamically adjusted per chunk based on the median of observed update norms
    for that specific chunk. Each chunk maintains its own threshold that evolves independently.
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
        tau_own=1.0,          # Norm threshold for own model update 
        tau_init=0.1,         # Initial norm threshold for neighbor updates
        update_window=5,      # Number of rounds before updating thresholds
    ):
        """
        Constructor for chunk-specific norm clipping defense class.
        
        Parameters:
        -----------
        rank, machine_id, etc. : same as parent class
        tau_own : float
            Norm threshold for the node's own model update (never scaled down)
        tau_init : float 
            Initial norm threshold for incoming neighbor updates, per chunk
        update_window : int
            Number of rounds between threshold updates
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
        
        self.tau_own = tau_own
        self.tau_init = tau_init
        self.update_window = update_window
        
        # For tracking norm history per chunk
        self.norm_history = defaultdict(list)
        self.thresholds = defaultdict(lambda: tau_init)  # Default threshold per chunk
        self.last_threshold_update = 0
        
        # For logging threshold changes
        self.threshold_history = defaultdict(list)
        for chunk_id in self.thresholds:
            self.threshold_history[chunk_id].append((0, tau_init))

        # Keep track of unique chunks seen
        self.chunks_seen = set()

    def initialize_defense_data(self):
        """Initialize data structures for chunk-specific norm clipping defense"""
        self.defense_data = {
            'neighbor_weights': defaultdict(list),      # Map node -> weight tensor
            'param_adversarial_sources': defaultdict(list),  # Track adversarial sources
            'round_norms': defaultdict(list)            # Track norms per chunk in current round
        }
        
    def defender_forward_averaging(self, data):
        """
        Process incoming updates and apply norm clipping to weight differences.
        The weight difference (update) between the neighbor's weights and our weights
        is clipped if its norm exceeds the threshold for that specific chunk.
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

        # Apply norm clipping to weight differences (updates) from neighbors
        if sender_node != self.uid:
            # Get chunk ID - using start index of the chunk as identifier
            chunk_id = data.get("start_index", indices[0].item())
            self.chunks_seen.add(chunk_id)
            
            # Get the corresponding chunk of our own weights using indices
            own_chunk = torch.index_select(self.defense_data['neighbor_weights'][self.uid], 0, indices)
            
            # Compute weight difference (update) from our current weights
            weight_diff = deserializedT - own_chunk

            # Compute norm of the difference and clip if needed
            norm = torch.norm(weight_diff, p=2).item()
            
            # Store norm for threshold adjustment, per chunk
            self.defense_data['round_norms'][chunk_id].append(norm)

            # Get threshold for this specific chunk
            threshold = self.thresholds[chunk_id]
            
            if norm > threshold:
                # Scale down the difference to have norm = threshold for this chunk
                scale_factor = threshold / norm
                weight_diff = weight_diff * scale_factor
                # Reconstruct weights from clipped difference
                deserializedT = own_chunk + weight_diff
        
        # Store the clipped chunk using scatter_
        self.defense_data['neighbor_weights'][sender_node].scatter_(0, indices, deserializedT)
        
        # Track adversarial sources 
        sender_is_adversarial = sender_node in self.adversarial_nodes
        self.defense_data['param_adversarial_sources'][sender_node] = sender_is_adversarial

    def _update_thresholds(self):
        """
        Update the clipping thresholds for each chunk based on its observed norm history
        """
        for chunk_id in self.chunks_seen:
            norms = self.norm_history[chunk_id]
            if len(norms) > 0:
                # Compute median of observed norms for this chunk
                median_norm = np.median(norms)
                
                # Set new threshold to median norm for this chunk
                self.thresholds[chunk_id] = median_norm
                
                # Log threshold change
                self.threshold_history[chunk_id].append((self.communication_round, median_norm))
                
                # Reset this chunk's norm history
                self.norm_history[chunk_id] = []
        
        self.last_threshold_update = self.communication_round

    def get_defended_model(self):
        """
        Return the averaged model after applying chunk-specific norm clipping defense.
        Also updates the dynamic thresholds if needed.
        """
        # First update the norm history with norms from this round
        if self.defense_data and 'round_norms' in self.defense_data:
            for chunk_id, norms in self.defense_data['round_norms'].items():
                self.norm_history[chunk_id].extend(norms)
                print(f"Node {self.uid} chunk {chunk_id}: {len(norms)} norms this round, {len(self.norm_history[chunk_id])} total")

        # Check if it's time to update the thresholds
        rounds_since_update = self.communication_round - self.last_threshold_update
        if rounds_since_update >= self.update_window and self.communication_round > 0:
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
        
        # Save threshold data if configured
        self._save_threshold_history()
        
        return state_dict

    def _save_threshold_history(self):
        """
        Save threshold history to a file for analysis
        """
        try:
            import json
            import os
            
            threshold_file = os.path.join(self.log_dir, f"chunk_thresholds_{self.uid}.json")
            
            # Convert defaultdict to regular dict for JSON serialization
            threshold_history_dict = {str(k): v for k, v in self.threshold_history.items()}
            thresholds_dict = {str(k): v for k, v in self.thresholds.items()}
            
            with open(threshold_file, 'w') as f:
                json.dump({
                    'threshold_history': threshold_history_dict,
                    'current_thresholds': thresholds_dict,
                    'initial_threshold': self.tau_init,
                    'update_window': self.update_window,
                    'chunks_seen': list(self.chunks_seen)
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving threshold history: {e}")

    def _cleanup_defense_data(self):
        """Clean up defense data structures between rounds"""
        # Keep the norm history but clear the defense data
        if self.defense_data:
            self.defense_data = None