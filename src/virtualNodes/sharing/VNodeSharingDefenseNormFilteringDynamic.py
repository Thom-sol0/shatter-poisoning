import torch
import numpy as np
from collections import defaultdict
import os
import json
from virtualNodes.sharing.VNodeSharingDefenseWeightTracker import VNodeSharingDefenseWeightTracker as VNodeSharingDefenseBase

class VNodeSharingPoison(VNodeSharingDefenseBase):
    """
    Dynamic Norm Filtering defense implementation.
    
    Cette classe implémente un mécanisme de défense par filtrage dynamique où chaque nœud honnête
    ignore complètement les mises à jour des voisins qui dépassent un seuil de norme qui évolue
    dynamiquement. Contrairement au norm clipping qui redimensionne les mises à jour excessives,
    cette approche exclut complètement les mises à jour malveillantes du processus d'agrégation,
    tout en adaptant le seuil basé sur la médiane des normes observées.
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
        save_interval=10,  # Sauvegarde des poids tous les N tours de communication
        experiment_id=None,  # Identifiant unique pour cette expérience
        tau_init=0.1,  # Seuil de norme initial pour les mises à jour des voisins
        update_window=5,  # Nombre de tours avant la mise à jour du seuil
    ):
        """
        Constructeur pour la classe de défense par filtrage de norme dynamique.
        
        Paramètres:
        -----------
        rank, machine_id, etc. : identiques à la classe parente
        tau_init : float 
            Seuil de norme initial pour les mises à jour des voisins
        update_window : int
            Nombre de tours entre les mises à jour du seuil
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
        
        self.tau_nbr = tau_init  # Seuil actuel, initialisé à tau_init
        self.tau_init = tau_init  # Conserver la valeur initiale comme référence
        self.update_window = update_window
        
        # Pour le suivi de l'historique des normes
        self.norm_history = []
        self.last_threshold_update = 0
        
        # Pour l'enregistrement des changements de seuil
        self.threshold_history = [(0, tau_init)]
        
        # Pour le suivi des nœuds filtrés
        self.filtered_nodes = set()
        self.filtered_stats = defaultdict(int)  # Compte combien de fois chaque nœud est filtré

    def initialize_defense_data(self):
        """Initialise les structures de données pour la défense par filtrage dynamique"""
        self.defense_data = {
            'neighbor_weights': defaultdict(list),  # Map nœud -> tenseur de poids
            'param_adversarial_sources': defaultdict(list),  # Suivi des sources adverses
            'round_norms': [],  # Suivi des normes observées dans le tour actuel
            'valid_neighbors': set()  # Suivi des nœuds avec mises à jour valides (en dessous du seuil)
        }
        self.filtered_nodes = set()
        
    def defender_forward_averaging(self, data):
        """
        Traite les mises à jour entrantes et filtre celles dont les normes dépassent le seuil.
        Au lieu de réduire les mises à jour à normes élevées, nous les ignorons complètement
        dans la moyenne.
        """
        # Initialise les données de défense si ce n'est pas fait
        if self.defense_data is None:
            self.initialize_defense_data()
            
            # Stocke d'abord les poids de son propre modèle
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            own_weights = torch.cat(tensors_to_cat, dim=0).to(self.device)
            self.defense_data['neighbor_weights'][self.uid] = own_weights
            self.defense_data['valid_neighbors'].add(self.uid)
            
        # Traite les données reçues
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
            
        # Initialise les poids du voisin s'ils n'ont pas encore été vus
        if sender_node not in self.defense_data['neighbor_weights']:
            self.defense_data['neighbor_weights'][sender_node] = self.defense_data['neighbor_weights'][self.uid].clone()

        # Évalue et filtre potentiellement les mises à jour des voisins
        if sender_node != self.uid:
            # Obtient la portion correspondante de nos propres poids en utilisant les indices
            own_chunk = torch.index_select(self.defense_data['neighbor_weights'][self.uid], 0, indices)
            # Calcule la différence de poids (mise à jour) par rapport à nos poids actuels
            weight_diff = deserializedT - own_chunk

            # Calcule la norme de la différence
            norm = torch.norm(weight_diff, p=2).item()
            
            # Stocke la norme pour l'ajustement du seuil
            self.defense_data['round_norms'].append(norm)

            # Si la norme dépasse le seuil, marque ce voisin comme non valide et n'utilise pas la mise à jour
            if norm > self.tau_nbr:
                # Suit les nœuds filtrés pour la journalisation/analyse
                self.filtered_nodes.add(sender_node)
                self.filtered_stats[sender_node] += 1
                
                print(f"Node {self.uid} filtered update from {sender_node} with norm {norm:.4f} > threshold {self.tau_nbr:.4f}")
                
                # Saute le stockage de cette mise à jour - nous conservons l'état existant
                # qui serait nos propres poids (première mise à jour) ou dernière mise à jour valide
                return
            else:
                # Marque ce voisin comme valide puisque sa mise à jour est en dessous du seuil
                self.defense_data['valid_neighbors'].add(sender_node)
        
        # Stocke la mise à jour uniquement si elle a passé le filtrage
        self.defense_data['neighbor_weights'][sender_node].scatter_(0, indices, deserializedT)
        
        # Suit les sources adverses
        sender_is_adversarial = sender_node in self.adversarial_nodes
        self.defense_data['param_adversarial_sources'][sender_node] = sender_is_adversarial

    def _update_threshold(self):
        """
        Mise à jour du seuil de filtrage basé sur l'historique des normes observées
        """
        if len(self.norm_history) > 0:
            # Calcule la médiane de toutes les normes observées
            median_norm = np.median(self.norm_history)
            old_threshold = self.tau_nbr
            
            # Définit le nouveau seuil comme étant la norme médiane
            self.tau_nbr = median_norm
            
            # Enregistre le changement de seuil
            self.threshold_history.append((self.communication_round, median_norm))
            
            # Log de la mise à jour vers la console
            print(f"Node {self.uid}: Updated tau_nbr from {old_threshold:.4f} to {median_norm:.4f} at round {self.communication_round}")
            
        # Réinitialise l'historique des normes
        self.norm_history = []
        self.last_threshold_update = self.communication_round

    def get_defended_model(self):
        """
        Renvoie le modèle moyenné en utilisant uniquement les mises à jour valides des voisins.
        Met également à jour le seuil dynamique si nécessaire.
        """
        # D'abord, met à jour l'historique des normes avec les normes de ce tour
        if 'round_norms' in self.defense_data:
            self.norm_history.extend(self.defense_data['round_norms'])
        
        # Vérifie s'il est temps de mettre à jour le seuil
        rounds_since_update = self.communication_round - self.last_threshold_update
        if rounds_since_update >= self.update_window and self.communication_round > 0:
            self._update_threshold()
            
        # Obtient uniquement les mises à jour des voisins valides qui n'ont pas été filtrées
        valid_neighbor_nodes = self.defense_data['valid_neighbors']
        valid_updates = [self.defense_data['neighbor_weights'][node] 
                         for node in valid_neighbor_nodes]
        
        # Statistiques de filtrage du log
        filtered_count = len(self.filtered_nodes)
        total_neighbors = len(set(self.neighbor_list))
        print(f"Node {self.uid}: Filtered {filtered_count}/{total_neighbors} neighbors due to high update norms")
        
        if not valid_updates:
            print(f"Warning: No valid updates found for node {self.uid}. Using own weights.")
            # Utilise ses propres poids si toutes les autres mises à jour ont été filtrées
            defended_weights = self.defense_data['neighbor_weights'][self.uid]
        else:
            # Moyenne simple des mises à jour valides uniquement
            defended_weights = torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
            for weights in valid_updates:
                defended_weights += weights
            defended_weights = defended_weights / len(valid_updates)
        
        # Nettoie les éventuelles valeurs NaN/Inf restantes
        if torch.any(torch.isnan(defended_weights)) or torch.any(torch.isinf(defended_weights)):
            defended_weights = self._detect_and_sanitize_nan_inf(
                defended_weights,
                "defended_weights",
                f"node_{self.uid}"
            )
            
        # Convertit en state dict
        state_dict = self._post_step(defended_weights)
        state_dict, was_corrupted = self._validate_model_state(state_dict, f"defended_model_node_{self.uid}")
        
        # Sauvegarde les données de seuil et de filtrage si configuré
        self._save_filtering_stats()
        
        return state_dict

    def _save_filtering_stats(self):
        """
        Sauvegarde l'historique du seuil et les statistiques de filtrage dans un fichier pour analyse
        """
        try:
            import json
            import os
            
            filtering_file = os.path.join(self.log_dir, f"filtering_stats_{self.uid}.json")
            
            # Convertit les ensembles en listes pour la sérialisation JSON
            filtered_nodes_list = list(self.filtered_nodes)
            
            # Calcule les statistiques sur les normes si disponibles
            norm_stats = {}
            if self.norm_history:
                norm_stats = {
                    'min': min(self.norm_history),
                    'max': max(self.norm_history),
                    'mean': sum(self.norm_history) / len(self.norm_history),
                    'median': sorted(self.norm_history)[len(self.norm_history)//2],
                    'count': len(self.norm_history)
                }
            
            with open(filtering_file, 'w') as f:
                json.dump({
                    'threshold_history': self.threshold_history,
                    'current_threshold': self.tau_nbr,
                    'initial_threshold': self.tau_init,
                    'update_window': self.update_window,
                    'filtered_nodes': filtered_nodes_list,
                    'filtered_stats': self.filtered_stats,
                    'communication_round': self.communication_round,
                    'norm_stats': norm_stats
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving filtering stats: {e}")

    def _cleanup_defense_data(self):
        """Nettoie les structures de données de défense entre les tours"""
        # Conserve l'historique des normes mais efface les autres données de défense
        if self.defense_data and 'round_norms' in self.defense_data:
            self.defense_data = None
        # Ne réinitialise pas filtered_nodes et filtered_stats pour conserver les statistiques globales