import copy
import logging
import os
import torch
import numpy as np
import json
from virtualNodes.sharing.VNodeSharingRandom import VNodeSharing
from collections import defaultdict

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

    
    def _apply_model_poisoning(self):
        """
        Apply poisoning strategy for adversarial nodes
        """
        # Check if we should poison in this round (based on poison_after)
        if self.poison_after > 1 and self.communication_round % self.poison_after != 0:
            logging.debug(f"Node {self.uid} skipping poisoning in round {self.communication_round}")
            # Just apply the averaged model when poisoning is skipped
            self.model.load_state_dict(self._post_step(self.current_sum))
        
        logging.info(f"Node {self.uid} applying {self.attack_type} poisoning in round {self.communication_round}")
    
        if self.attack_type == 'zero':
            # Zero out all parameters
            self.current_sum.zero_()
            self.model.load_state_dict(self._post_step(self.current_sum))
        
        elif self.attack_type == 'flip_grad':
            # Mirror the gradient update to move away from consensus
            mean_state_dict = self._post_step(self.current_sum)
            init_state_dict = copy.deepcopy(self.model.state_dict())
            final_state_dict = {}
            for key in init_state_dict:
                final_state_dict[key] = 2 * init_state_dict[key] - mean_state_dict[key]
            self.model.load_state_dict(final_state_dict)
        
        else:
            logging.warning(f"Unknown attack type: {self.attack_type}")
            # Use normal averaged model for unknown attack types
            self.model.load_state_dict(self._post_step(self.current_sum))

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
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        logging.debug(
            "Forward Averaging model from neighbor {} of iteration {}".format(
                data["vSource"], iteration
            )
        )
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        logging.debug("Deserialized model from neighbor {}".format(data["vSource"]))
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1

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
        if "degree" in data:
            del data["degree"]
        del data["iteration"]
        del data["CHANNEL"]
        logging.debug(
            "Forward Averaging model from neighbor {} of iteration {}".format(
                data["vSource"], iteration
            )
        )
    
        # Deserialize received model data
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
    
        logging.debug("Deserialized model from neighbor {}".format(data["vSource"]))
    
        # Add the received weights to the param_values dictionary
        for idx, value in zip(indices.tolist(), deserializedT.tolist()):
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

        self.current_weights = self.current_weights.type(torch.float32)
        self.current_weights = 1.0 / self.current_weights
        self.current_sum = self.current_sum * self.current_weights
        logging.debug("Finished averaging")
        self.current_sum = self.current_sum.cpu()

        self._apply_model_poisoning()
        
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
        for idx in range(self.total_length):
            if idx in self.param_values and len(self.param_values[idx]) > 0:
                median_weights[idx] = torch.tensor(
                    np.median(self.param_values[idx]),
                    dtype=torch.float32,
                    device=self.device
                )
    
        # Convert flat tensor back to model state dict
        return self._post_step(median_weights)

    def defender_finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging for defender nodes.
        Applies median-based defense against poisoning attacks.
        """
        # Process all incoming data
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        # Calculate the weight-wise median and update model
        medgrad_model = self.get_medgrad_model()
        self.model.load_state_dict(medgrad_model)
    
        logging.debug("Finished median-based averaging")
    
        # Clean up for the next round
        self.communication_round += 1
        self.param_values = None

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """
        if self.uid in self.adversarial_nodes:
            self.adversarial_finish_forward_averaging(peer_deques)
        else:
            self.defender_finish_forward_averaging(peer_deques)





        

