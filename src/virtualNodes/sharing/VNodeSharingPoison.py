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
        sigma=0.1,  # Standard deviation for random noise
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
        if attack_type not in ['zero', 'random']:
            raise ValueError(f"Unsupported attack type: {attack_type}. Supported types: ['zero', 'random']")
        if attack_type == 'random':
            self.sigma = sigma
        self.poison_after = int(poison_after) if poison_after is not None else 1
        
        if isinstance(adversarial_nodes, str) and adversarial_nodes:
            self.adversarial_nodes = [int(node_id.strip()) for node_id in adversarial_nodes.split(',')]
        else:
            self.adversarial_nodes = [] if adversarial_nodes is None else adversarial_nodes

        #for node_id in self.adversarial_nodes:
        #    if node_id >= self.graph.num_nodes:
        #        raise ValueError(f"Adversarial node ID {node_id} exceeds number of nodes in the graph.")

        # self.log_poisoning_metrics = bool(log_poisoning_metrics)
        
        # self.poison_metrics = {
        #     "rounds_poisoned": 0,
        #     "total_messages": 0,
        #     "poisoned_messages": 0
        # }
        
        # logging.info(f"Node {rank} initialized with {attack_type} poisoning attack")
    
    def _apply_poison(self, params):
        """Apply poisoning strategy to sent data (gradients)"""
        with torch.no_grad():
            if self.attack_type == 'zero':
                params.zero_()
        # TODO: Implement other poisoning strategies
        return params
    
    def get_data_to_send(self, vnodes_per_node=1, degree=None, sparsity=0.0):
        self._pre_step()
        data_list = self.serialized_models(
            vnodes_per_node=vnodes_per_node
        )
        for data in data_list:
            # Apply poisoning if this node is adversarial
            if self.uid in self.adversarial_nodes and self.communication_round % self.poison_after == 0:
                data = self._apply_poison(data)
            data["real_node"] = self.uid
        return data_list
    
    # def _log_poison_metrics(self):
    #     """Log poisoning metrics to file"""
    #     metrics_path = os.path.join(self.log_dir, f"poison_metrics_{self.uid}.json")
        
    #     if self.poison_metrics["total_messages"] > 0:
    #         poison_rate = self.poison_metrics["poisoned_messages"] / self.poison_metrics["total_messages"]
    #         self.poison_metrics["poison_rate"] = poison_rate
        
    #     with open(metrics_path, 'w') as f:
    #         json.dump(self.poison_metrics, f, indent=2)
            
    # def __del__(self):
    #     """Save final metrics before destruction"""
    #     if self.log_poisoning_metrics:
    #         self._log_poison_metrics()

    def _get_poisoned_model(self, deserializedT):
        """
        Applies the poisoning
        """
        poisonedT = copy.deepcopy(deserializedT)
    
        if self.attack_type == 'zero':
            # Zero out all parameters
            for key in poisonedT:
                poisonedT[key].zero_()
            
        elif self.attack_type == 'random':
            # Add normalized random noise
            for key in poisonedT:
                noise = torch.randn_like(poisonedT[key])
                poisonedT[key] += 0.1 * noise / torch.norm(noise)
    
        return poisonedT
    


    def _post_step(self, T):
        """
        Return state_dict of model.

        Parameters
        ----------
        T : torch.Tensor
            Flat model vector

        Returns
        -------
        state_dict
            state_dict of model

        """
        state_dict = self.model.state_dict()
        start_index = 0
        for i, key in enumerate(state_dict):
            end_index = start_index + self.lens[i]
            state_dict[key] = T[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index
        return state_dict

    def poison_forward_averaging(self, data):
        """
        Computes the poisoned sum for the average in a state based manner.

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
            self.current_weights = torch.zeros(
                self.total_length + 1, dtype=torch.float32, device=self.device
            )
            self.current_weights[0] = 1
            self.current_weights[-1] = -1

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
            deserializedT, start, end = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        logging.debug("Deserialized model from neighbor {}".format(data["vSource"]))
        poisonedT = self._get_poisoned_model(deserializedT)
        self.current_sum[start:end] += poisonedT.to(self.device)
        self.current_weights[start] += 1
        self.current_weights[end] -= 1
    


    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """

        # if self.uid in self.adversarial_nodes and self.communication_round % self.poison_after == 0:
        #     # If the node is adversarial, instead of doing averaging aggregation,
        #     # it will just add to the current model
        #     logging.debug(f"Node {self.uid} is adversarial, applying poisoning")
        # 
        # if self.uid in self.adversarial_nodes and self.communication_round % self.poison_after == 0:
        #     for _, n in enumerate(peer_deques):
        #         for data in peer_deques[n]:
        #             # If the node is adversarial, apply poisoning
        #             self.poison_forward_averaging(data)
        # else: 
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                # Otherwise, do normal forward averaging
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        self.current_weights = torch.cumsum(self.current_weights, dim=0)[:-1]
        self.current_weights = self.current_weights.type(torch.float32)
        self.current_weights = 1.0 / self.current_weights
        self.current_sum = self.current_sum * self.current_weights
        logging.debug("Finished averaging")
        self.current_sum = self.current_sum.cpu()
        self.model.load_state_dict(self._post_step(self.current_sum))
        self.communication_round += 1
        self.current_weights = None
        self.current_sum = None