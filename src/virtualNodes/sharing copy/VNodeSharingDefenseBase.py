import copy
import logging
import os
import torch
import numpy as np
import json
from virtualNodes.sharing.VNodeSharingRandom import VNodeSharing
from collections import defaultdict
import warnings
from abc import ABC, abstractmethod

class VNodeSharingDefenseBase(VNodeSharing, ABC):
    """
    Base class for poisoned model sharing with defense mechanisms
    Implements common poisoning strategies and provides abstract methods for defenses
    """

    def _parse_adversarial_nodes(self, adversarial_nodes):
        """
        Parse the adversarial nodes input to ensure it's a list of integers.
        Handles string input, single integers, and lists.
        """
        if isinstance(adversarial_nodes, str) and adversarial_nodes:
            return [int(node_id.strip()) for node_id in adversarial_nodes.split(',') if node_id.strip()]
        elif adversarial_nodes is None:
            return []
        elif not isinstance(adversarial_nodes, list):
            return [adversarial_nodes]
        else:
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
        adversarial_nodes=None,
        poison_after=None,
    ):
        """
        Constructor for defense base class
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

        # Defense-specific data structures (to be used by concrete implementations)
        self.defense_data = None
        
        # Diagnostic tracking for NaN/Inf issues
        self.nan_inf_detection_count = 0
        self.corrupted_weights_received = 0
        self.corrupted_weights_rejected = 0
        
        # Track adversarial proportion for each weight position
        self.param_adversarial_sources = None
        self.max_adversarial_proportion = 0.0

    def _detect_and_sanitize_nan_inf(self, tensor, tensor_name="tensor", sender_node="unknown", default_value=0.0):
        """
        Detect and sanitize NaN/Inf values in tensors.
        """
        if tensor is None:
            return torch.zeros_like(tensor) if tensor is not None else None
        
        nan_mask = torch.isnan(tensor)
        nan_count = torch.sum(nan_mask).item()
        
        inf_mask = torch.isinf(tensor)
        inf_count = torch.sum(inf_mask).item()
        
        large_mask = torch.abs(tensor) > 1e10
        large_count = torch.sum(large_mask).item()
        
        total_corrupted = nan_count + inf_count
        
        if total_corrupted > 0:
            self.nan_inf_detection_count += 1
            self.corrupted_weights_received += total_corrupted
            
            sanitized_tensor = tensor.clone()
            
            if nan_count > 0:
                sanitized_tensor[nan_mask] = default_value
            
            if inf_count > 0:
                pos_inf_mask = torch.isposinf(tensor)
                neg_inf_mask = torch.isneginf(tensor)
                sanitized_tensor[pos_inf_mask] = 1e6
                sanitized_tensor[neg_inf_mask] = -1e6
            
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
                
                sanitized_tensor = self._detect_and_sanitize_nan_inf(tensor, f"layer_{layer_name}", source)
                state_dict[layer_name] = sanitized_tensor
        
        return state_dict, len(corrupted_layers) > 0

    def _apply_model_poisoning(self):
        """
        Apply poisoning strategy for adversarial nodes
        """
        if self.poison_after > 1 and self.communication_round % self.poison_after != 0:
            mean_state_dict = self._post_step(self.current_sum)
            mean_state_dict, was_corrupted = self._validate_model_state(mean_state_dict, f"mean_averaging_node_{self.uid}")
            self.model.load_state_dict(mean_state_dict)
            return
    
        if self.attack_type == 'zero':
            self.current_sum.zero_()
            zero_state_dict = self._post_step(self.current_sum)
            zero_state_dict, was_corrupted = self._validate_model_state(zero_state_dict, f"zero_attack_node_{self.uid}")
            self.model.load_state_dict(zero_state_dict)
        
        elif self.attack_type == 'flip_grad':
            mean_state_dict = self._post_step(self.current_sum)
            init_state_dict = copy.deepcopy(self.model.state_dict())
            
            mean_state_dict, mean_corrupted = self._validate_model_state(mean_state_dict, f"mean_state_node_{self.uid}")
            init_state_dict, init_corrupted = self._validate_model_state(init_state_dict, f"init_state_node_{self.uid}")
            
            final_state_dict = {}
            
            for key in init_state_dict:
                try:
                    flipped_update = 2 * init_state_dict[key] - mean_state_dict[key]
                    
                    if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                        flipped_update = self._detect_and_sanitize_nan_inf(
                            flipped_update, 
                            f"flip_grad_{key}", 
                            f"node_{self.uid}",
                            default_value=0.0
                        )
                        
                        if torch.any(torch.isnan(flipped_update)) or torch.any(torch.isinf(flipped_update)):
                            final_state_dict[key] = mean_state_dict[key]
                        else:
                            final_state_dict[key] = flipped_update
                    else:
                        if torch.max(torch.abs(flipped_update)) > 1e8:
                            flipped_update = torch.clamp(flipped_update, min=-1e8, max=1e8)
                        final_state_dict[key] = flipped_update
                        
                except Exception as e:
                    final_state_dict[key] = mean_state_dict[key]
            
            final_state_dict, final_corrupted = self._validate_model_state(final_state_dict, f"final_flip_grad_node_{self.uid}")
            self.model.load_state_dict(final_state_dict)
        
        else:
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
        for data in data_list:
            data["real_node"] = self.uid
        return data_list

    def forward_averaging(self, data):
        """
        Route to appropriate averaging method based on node type
        """
        if self.uid in self.adversarial_nodes:
            self.adversarial_forward_averaging(data)
        else:
            self.defender_forward_averaging(data)

    def adversarial_forward_averaging(self, data):
        """
        Standard averaging for adversarial nodes (same for all defenses)
        """
        if self.current_sum == None:
            self.current_weights = (
                torch.zeros(self.total_length, dtype=torch.float32, device=self.device) + 1
            )

            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            self.current_sum = torch.cat(tensors_to_cat, dim=0).to(self.device)

        iteration = data["iteration"]
        sender_node = data.get("vSource", "unknown")
        real_node_id = data.get("real_node", None)
        
        # Clean up data
        for key in ["degree", "iteration", "CHANNEL", "real_node"]:
            if key in data:
                del data[key]
        
        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        
        if torch.any(torch.isnan(deserializedT)) or torch.any(torch.isinf(deserializedT)):
            deserializedT = self._detect_and_sanitize_nan_inf(
                deserializedT, 
                f"received_weights_from_{sender_node}", 
                sender_node
            )
        
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_before_update", 
                f"node_{self.uid}"
            )
        
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1
        
        if torch.any(torch.isnan(self.current_sum)) or torch.any(torch.isinf(self.current_sum)):
            self.current_sum = self._detect_and_sanitize_nan_inf(
                self.current_sum, 
                "current_sum_after_update", 
                f"node_{self.uid}"
            )

    @abstractmethod
    def defender_forward_averaging(self, data):
        """
        Abstract method for defender forward averaging - implement in subclasses
        """
        pass

    @abstractmethod
    def get_defended_model(self):
        """
        Abstract method to get the defended model - implement in subclasses
        This should return a model state dictionary
        """
        pass

    @abstractmethod
    def initialize_defense_data(self):
        """
        Abstract method to initialize defense-specific data structures
        """
        pass

    def adversarial_finish_forward_averaging(self, peer_deques):
        """
        Standard finish for adversarial nodes (same for all defenses)
        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

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
                default_value=1.0
            )

        self.current_weights = self.current_weights.type(torch.float32)
        
        zero_weights_mask = self.current_weights == 0
        if torch.any(zero_weights_mask):
            self.current_weights[zero_weights_mask] = 1.0
        
        self.current_weights = 1.0 / self.current_weights
        
        if torch.any(torch.isnan(self.current_weights)) or torch.any(torch.isinf(self.current_weights)):
            self.current_weights = self._detect_and_sanitize_nan_inf(
                self.current_weights, 
                "current_weights_after_division", 
                f"node_{self.uid}",
                default_value=1.0
            )
        
        self.current_sum = self.current_sum * self.current_weights
        
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

    def defender_finish_forward_averaging(self, peer_deques):
        """
        Standard finish for defender nodes - uses abstract defended model
        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        # Get the defended model using the specific defense mechanism
        defended_model = self.get_defended_model()
        self.model.load_state_dict(defended_model)
    
        self._save_corruption_metrics()
    
        # Clean up for the next round
        self.communication_round += 1
        self._cleanup_defense_data()
        self.nan_inf_detection_count = 0
        self.corrupted_weights_received = 0
        self.corrupted_weights_rejected = 0

    @abstractmethod
    def _cleanup_defense_data(self):
        """
        Abstract method to clean up defense-specific data structures
        """
        pass

    def finish_forward_averaging(self, peer_deques):
        """
        Route to appropriate finish method based on node type
        """
        if self.uid in self.adversarial_nodes:
            self.adversarial_finish_forward_averaging(peer_deques)
        else:
            self.defender_finish_forward_averaging(peer_deques)

    # Common utility methods (same for all defenses)
    def compute_max_adversarial_proportion(self):
        """
        Compute the maximum proportion of adversarial values across all weight positions.
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
                    
                    weight_stats[idx] = {
                        'total_values': total_values,
                        'adversarial_count': adversarial_count,
                        'proportion': proportion,
                        'adversarial_sources': [node for node, is_adv in sources if is_adv]
                    }
        
        if adversarial_proportions:
            max_proportion = max(adversarial_proportions)
            avg_proportion = sum(adversarial_proportions) / len(adversarial_proportions)
            
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

    def _save_corruption_metrics(self):
        """
        Save corruption metrics to file
        """
        try:
            max_adv_prop, adv_stats = self.compute_max_adversarial_proportion()
            self.max_adversarial_proportion = max_adv_prop
            
            metrics_file = os.path.join(self.log_dir, f"corruption_metrics_{self.uid}.json")
            current_max_adv_prop = getattr(self, 'max_adversarial_proportion', 0.0)
            
            existing_data = {}
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_data = {}
            
            iteration_key = str(self.communication_round)
            
            if iteration_key in existing_data:
                existing_value = existing_data[iteration_key]
                new_value = max(existing_value, current_max_adv_prop)
            else:
                new_value = current_max_adv_prop
            
            existing_data[iteration_key] = new_value
            
            with open(metrics_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            pass