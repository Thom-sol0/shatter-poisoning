import torch
import numpy as np
from collections import defaultdict
from virtualNodes.sharing.VNodeSharingDefenseWeightTracker import VNodeSharingDefenseWeightTracker as VNodeSharingDefenseBase

class VNodeSharingDefenseNormFiltering(VNodeSharingDefenseBase):
    """
    Norm filtering defense implementation.
    
    This class implements a norm filtering defense mechanism where each honest node
    completely ignores updates from neighbors that exceed a specified norm threshold.
    Unlike norm clipping, which scales down excessive updates, this approach
    completely excludes malicious updates from the aggregation process.
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
        tau_nbr=0.1,  # Norm threshold for neighbor updates
    ):
        """
        Constructor for norm filtering defense class.
        
        Parameters:
        -----------
        rank, machine_id, etc. : same as parent class
        tau_nbr : float 
            Norm threshold for incoming neighbor updates (updates exceeding this will be ignored)
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
        
        self.tau_nbr = tau_nbr
        # Track which nodes are filtered out due to high norm updates
        self.filtered_nodes = set()

    def initialize_defense_data(self):
        """Initialize data structures for norm filtering defense"""
        self.defense_data = {
            'neighbor_weights': defaultdict(list),  # Map node -> weight tensor
            'param_adversarial_sources': defaultdict(list),  # Track adversarial sources
            'valid_neighbors': set()  # Track nodes with valid (below threshold) updates
        }
        self.filtered_nodes = set()
        
    def defender_forward_averaging(self, data):
        """
        Process incoming updates and filter out those with norms exceeding the threshold.
        Instead of clipping high-norm updates, we completely ignore them in the averaging.
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
            self.defense_data['valid_neighbors'].add(self.uid)
            
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

        # Evaluate and potentially filter out updates from neighbors
        if sender_node != self.uid:
            # Get the corresponding chunk of our own weights using indices
            own_chunk = torch.index_select(self.defense_data['neighbor_weights'][self.uid], 0, indices)
            # Compute weight difference (update) from our current weights
            weight_diff = deserializedT - own_chunk

            # Compute norm of the difference
            norm = torch.norm(weight_diff, p=2)

            # If norm exceeds threshold, mark this neighbor as invalid and don't use the update
            if norm > self.tau_nbr:
                # Track filtered nodes for logging/analysis
                self.filtered_nodes.add(sender_node)
                
                # Skip storing this update - instead, we'll leave the existing state
                # which would be our own weights (first update) or last valid update
                return
            else:
                # Mark this neighbor as valid since its update is below threshold
                self.defense_data['valid_neighbors'].add(sender_node)
        
        # Store the update only if it passed the filtering
        self.defense_data['neighbor_weights'][sender_node].scatter_(0, indices, deserializedT)
        
        # Track adversarial sources 
        sender_is_adversarial = sender_node in self.adversarial_nodes
        self.defense_data['param_adversarial_sources'][sender_node] = sender_is_adversarial

    def get_defended_model(self):
        """
        Return the averaged model using only valid (below threshold norm) neighbor updates.
        """
        # Get only the valid neighbor updates that weren't filtered out
        valid_neighbor_nodes = self.defense_data['valid_neighbors']
        valid_updates = [self.defense_data['neighbor_weights'][node] 
                         for node in valid_neighbor_nodes]
        
        # Log filtering statistics
        filtered_count = len(self.filtered_nodes)
        total_neighbors = len(self.neighbor_list)
        print(f"Node {self.uid}: Filtered {filtered_count}/{total_neighbors} neighbors due to high update norms")
        
        if not valid_updates:
            print(f"Warning: No valid updates found for node {self.uid}. Using own weights.")
            # Use own weights if all other updates were filtered out
            defended_weights = self.defense_data['neighbor_weights'][self.uid]
        else:
            # Simple average of valid updates only
            defended_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
            for weights in valid_updates:
                defended_weights += weights
            defended_weights = defended_weights / len(valid_updates)
        
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
        
        return state_dict

    def _cleanup_defense_data(self):
        """Clean up defense data structures between rounds"""
        self.defense_data = None
        self.filtered_nodes = set()
