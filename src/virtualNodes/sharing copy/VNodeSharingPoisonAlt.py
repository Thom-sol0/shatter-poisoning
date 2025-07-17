import copy
import logging
import os
import torch
import numpy as np
import json
from virtualNodes.sharing.VNodeSharingRandom import VNodeSharing

class VNodeSharingPoison(VNodeSharing):
    """
    Poisoned model sharing class that sends malicious gradients
    Implements various poisoning strategies for adversarial attacks
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
        
        # Safety parameters for preventing crashes
        self.eps = 1e-8  # Small epsilon for numerical stability
        
        if isinstance(adversarial_nodes, str) and adversarial_nodes:
            # Parse string representation
            self.adversarial_nodes = [int(node_id.strip()) for node_id in adversarial_nodes.split(',') if node_id.strip()]
        elif adversarial_nodes is None:
            # No adversarial nodes
            self.adversarial_nodes = []
        elif not isinstance(adversarial_nodes, list):
            # Single integer or other non-list type
            self.adversarial_nodes = [adversarial_nodes]
        else:
            # Already a list
            self.adversarial_nodes = adversarial_nodes

        print(f"Adversarial nodes: {self.adversarial_nodes}")

    
    def get_data_to_send(self, vnodes_per_node=1, degree=None, sparsity=0.0):
        self._pre_step()
        data_list = self.serialized_models(
            vnodes_per_node=vnodes_per_node, sparsity=sparsity
        )
        return data_list
    

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        self.current_weights = self.current_weights.type(torch.float32)
        self.current_weights = 1.0 / self.current_weights
        self.current_sum = self.current_sum * self.current_weights
        logging.debug("Finished averaging")
        self.current_sum = self.current_sum.cpu()

        if self.uid not in self.adversarial_nodes:
           self.model.load_state_dict(self._post_step(self.current_sum))
        if self.attack_type == 'zero' and self.uid in self.adversarial_nodes:
            self.current_sum.zero_()
            self.model.load_state_dict(self._post_step(self.current_sum))
        elif self.attack_type == 'flip_grad' and self.uid in self.adversarial_nodes:
            mean_state_dict = self._post_step(self.current_sum)
            init_state_dict = copy.deepcopy(self.model.state_dict())
            final_state_dict = {}
            
            for key in init_state_dict.keys():
                # Calculate the flipped update: 2 * init - mean
                flipped_update = 2 * init_state_dict[key] - mean_state_dict[key]
                
                # Check for and fix numerical issues
                if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                    logging.warning(f"Node {self.uid}: NaN/Inf detected in {key} during flip_grad, using mean state instead")
                    final_state_dict[key] = mean_state_dict[key]
                else:
                    final_state_dict[key] = flipped_update
            
            self.model.load_state_dict(final_state_dict)

        self.communication_round += 1
        self.current_weights = None
        self.current_sum = None






