import torch
from virtualNodes.sharing.VNodeSharingDefenseBase import VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    No defense implementation - replicates VNodeSharingPoisonAlt behavior exactly
    """

    def initialize_defense_data(self):
        """
        No special initialization needed for no defense
        """
        pass

    def defender_forward_averaging(self, data):
        """
        Standard forward averaging - same as adversarial nodes (no defense)
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

        try:
            deserializedT, indices = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e

        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1

    def get_defended_model(self):
        """
        Standard averaging (no defense) - exactly like VNodeSharingPoisonAlt
        """
        assert self.current_sum != None
        assert self.current_weights != None

        self.current_weights = self.current_weights.type(torch.float32)
        self.current_weights = 1.0 / self.current_weights
        self.current_sum = self.current_sum * self.current_weights
        self.current_sum = self.current_sum.cpu()

        return self._post_step(self.current_sum)

    def _cleanup_defense_data(self):
        """
        Clean up - same as VNodeSharingPoisonAlt
        """
        self.current_weights = None
        self.current_sum = None