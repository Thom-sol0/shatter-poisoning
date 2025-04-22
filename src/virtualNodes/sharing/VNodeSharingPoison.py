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
        poison_strength=1.0,
        poison_probability=1.0,
        targeted_nodes=None,
        log_poisoning_metrics=True,
    ):
        """
        Constructor for poisoning class
        
        Parameters
        ----------
        rank, machine_id, etc. : same as parent class
        attack_type : str
            Poisoning strategy ('zero', 'flip', 'noise', 'scale')
        poison_strength : float
            Strength of the poisoning attack (scaling factor for flip/noise)
        poison_probability : float
            Probability of poisoning a message (0.0-1.0)
        targeted_nodes : list
            List of specific node UIDs to target (None = attack all)
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
        self.poison_strength = float(poison_strength)
        self.poison_probability = float(poison_probability)
        self.targeted_nodes = [] if targeted_nodes is None else targeted_nodes
        self.log_poisoning_metrics = bool(log_poisoning_metrics)
        
        self.poison_metrics = {
            "rounds_poisoned": 0,
            "total_messages": 0,
            "poisoned_messages": 0,
            "targeted_nodes": []
        }
        
        logging.info(f"Node {rank} initialized with {attack_type} poisoning attack (strength={poison_strength})")
    
    def _apply_poison(self, params):
        """Apply poisoning strategy to sent data (gradients)"""
        with torch.no_grad():
            if self.attack_type == 'zero':
                params.zero_()
            elif self.attack_type == 'flip':
                params.mul_(-self.poison_strength)
            elif self.attack_type == 'noise':
                noise = torch.randn_like(params) * self.poison_strength
                params.add_(noise)
            elif self.attack_type == 'scale':
                params.mul_(self.poison_strength)
        return params
    
    def get_data_to_send(self, vnodes_per_node=1, sparsity=0.0, degree=None):
        """Override get_data_to_send to apply poisoning to outgoing data"""
        self._pre_step()
        data_list = self.serialized_models(
            vnodes_per_node=vnodes_per_node, sparsity=sparsity
        )
        
        self.poison_metrics["total_messages"] += len(data_list)
        
        for data in data_list:
            do_poison = np.random.random() < self.poison_probability
            
            if do_poison:
                data['params'] = self._apply_poison(data['params'])
                data['poisoned'] = True
                self.poison_metrics["poisoned_messages"] += 1
            else:
                data['poisoned'] = False
                
            data["real_node"] = self.uid
        
        self.poison_metrics["rounds_poisoned"] += 1
        
        if self.log_poisoning_metrics and self.poison_metrics["rounds_poisoned"] % 10 == 0:
            self._log_poison_metrics()
            
        return data_list
    
    def _log_poison_metrics(self):
        """Log poisoning metrics to file"""
        metrics_path = os.path.join(self.log_dir, f"poison_metrics_{self.uid}.json")
        
        if self.poison_metrics["total_messages"] > 0:
            poison_rate = self.poison_metrics["poisoned_messages"] / self.poison_metrics["total_messages"]
            self.poison_metrics["poison_rate"] = poison_rate
        
        with open(metrics_path, 'w') as f:
            json.dump(self.poison_metrics, f, indent=2)
            
    def __del__(self):
        """Save final metrics before destruction"""
        if self.log_poisoning_metrics:
            self._log_poison_metrics()