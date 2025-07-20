import torch
import numpy as np
from collections import defaultdict
from virtualNodes.sharing.VNodeSharingDefenseBase import VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    Norm clipping defense implementation.
    
    This class implements a norm clipping defense mechanism where each honest node
    limits the magnitude (Euclidean norm) of incoming neighbor updates. Updates from
    neighbors with norms exceeding a threshold are scaled down to prevent any single
    chunk from having outsized influence.
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
        tau_own=1.0,  # Norm threshold for own model update 
        tau_nbr=0.1,  # Norm threshold for neighbor updates
    ):
        """
        Constructor for norm clipping defense class.
        
        Parameters:
        -----------
        rank, machine_id, etc. : same as parent class
        tau_own : float
            Norm threshold for the node's own model update (never scaled down)
        tau_nbr : float 
            Norm threshold for incoming neighbor updates (scaled down if exceeded)
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
            poison_after=poison_after
        )
        
        self.tau_own = tau_own
        self.tau_nbr = tau_nbr

    def initialize_defense_data(self):
        """Initialize data structures for norm clipping defense"""
        self.defense_data = {
            'neighbor_weights': defaultdict(list),  # Map node -> weight tensor
            'param_adversarial_sources': defaultdict(list)  # Track adversarial sources
        }
        
    def defender_forward_averaging(self, data):
        """
        Process incoming updates and apply norm clipping to weight differences.
        The weight difference (update) between the neighbor's weights and our weights
        is clipped if its norm exceeds the threshold.
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
            # Get the corresponding chunk of our own weights using indices
            own_chunk = torch.index_select(self.defense_data['neighbor_weights'][self.uid], 0, indices)
            # Compute weight difference (update) from our current weights
            weight_diff = deserializedT - own_chunk

            # Compute norm of the difference and clip if needed
            norm = torch.norm(weight_diff, p=2)
            #print(f"Norm of weight difference from {sender_node}: {norm.item()}")
            if norm > self.tau_nbr:
                print(f"Clipping weight difference from {sender_node} with norm {norm.item()} to tau_nbr {self.tau_nbr}")
                # Scale down the difference to have norm = tau_nbr
                scale_factor = self.tau_nbr / norm
                weight_diff = weight_diff * scale_factor
                # Reconstruct weights from clipped difference
                deserializedT = own_chunk + weight_diff
        
        # Store the clipped chunk using scatter_
        self.defense_data['neighbor_weights'][sender_node].scatter_(0, indices, deserializedT)
        
        # Track adversarial sources 
        sender_is_adversarial = sender_node in self.adversarial_nodes
        self.defense_data['param_adversarial_sources'][sender_node] = sender_is_adversarial

    def get_defended_model(self):
        """
        Return the averaged model after applying norm clipping defense.
        """
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
        
        return state_dict

    def _cleanup_defense_data(self):
        """Clean up defense data structures between rounds"""
        self.defense_data = None
