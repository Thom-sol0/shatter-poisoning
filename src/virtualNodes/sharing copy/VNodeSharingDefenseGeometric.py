import torch
import numpy as np
from collections import defaultdict
from virtualNodes.sharing.VNodeSharingDefenseBase import VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    Geometric mean-based defense implementation
    """

    def initialize_defense_data(self):
        """
        Initialize geometric mean defense data structures
        """
        self.defense_data = {
            'param_values': defaultdict(list),
            'param_adversarial_sources': defaultdict(list)
        }

    def defender_forward_averaging(self, data):
        """
        Collect parameter values for geometric mean-based aggregation
        """
        if self.defense_data is None:
            self.initialize_defense_data()
            self.param_adversarial_sources = self.defense_data['param_adversarial_sources']
        
            # Add own model weights
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            own_weights = torch.cat(tensors_to_cat, dim=0).to(self.device)
        
            for i in range(len(own_weights)):
                self.defense_data['param_values'][i].append(own_weights[i].item())
                self.param_adversarial_sources[i].append((self.uid, False))

        # Process received data (same as median)
        iteration = data["iteration"]
        sender_node = data.get("real_node", data.get("vSource", "unknown"))
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
    
        sender_is_adversarial = sender_node in self.adversarial_nodes
        for idx, value in zip(indices.tolist(), deserializedT.tolist()):
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            self.defense_data['param_values'][idx].append(value)
            self.param_adversarial_sources[idx].append((sender_node, sender_is_adversarial))

    def get_defended_model(self):
        """
        Compute element-wise geometric mean for defense
        """
        geomean_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
    
        for idx in range(self.total_length):
            if idx in self.defense_data['param_values'] and len(self.defense_data['param_values'][idx]) > 0:
                values = self.defense_data['param_values'][idx]
                filtered_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                
                if len(filtered_values) > 0:
                    values_array = np.array(filtered_values)
                    
                    # Handle negative values and zeros for geometric mean
                    signs = np.sign(values_array)
                    abs_values = np.abs(values_array)
                    
                    # Replace zeros with small positive value
                    abs_values[abs_values == 0] = 1e-10
                    
                    # Compute geometric mean using log-space
                    log_mean = np.mean(np.log(abs_values))
                    geomean_abs = np.exp(log_mean)
                    
                    # Determine overall sign (majority vote)
                    positive_count = np.sum(signs > 0)
                    negative_count = np.sum(signs < 0)
                    
                    if positive_count >= negative_count:
                        geomean_val = geomean_abs
                    else:
                        geomean_val = -geomean_abs
                    
                    if np.isnan(geomean_val) or np.isinf(geomean_val):
                        geomean_val = 0.0
                    
                    geomean_weights[idx] = torch.tensor(
                        geomean_val,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    geomean_weights[idx] = 0.0
    
        if torch.any(torch.isnan(geomean_weights)) or torch.any(torch.isinf(geomean_weights)):
            geomean_weights = self._detect_and_sanitize_nan_inf(
                geomean_weights, 
                "geomean_weights", 
                f"node_{self.uid}"
            )
    
        state_dict = self._post_step(geomean_weights)
        state_dict, was_corrupted = self._validate_model_state(state_dict, f"geomean_model_node_{self.uid}")
        
        return state_dict

    def _cleanup_defense_data(self):
        """
        Clean up geometric mean defense data
        """
        self.defense_data = None
        self.param_adversarial_sources = None