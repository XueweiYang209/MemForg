"""
    Straight-forward LOSS attack, as described in https://ieeexplore.ieee.org/abstract/document/8429311
"""
import torch as ch
from evaluation.attacks.all_attacks import Attack
from evaluation.models import Model
from evaluation.config import ExperimentConfig


class LOSSAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)

    @ch.no_grad()
    def _attack(self, document, probs, **kwargs):
        """
            LOSS-score. Use log-likelihood from model.
        """
        return self.target_model.get_ll(document, probs=probs)
