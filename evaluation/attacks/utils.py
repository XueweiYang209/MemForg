from evaluation.attacks.all_attacks import AllAttacks

from evaluation.attacks.loss import LOSSAttack
from evaluation.attacks.reference import ReferenceAttack
from evaluation.attacks.zlib import ZLIBAttack
from evaluation.attacks.min_k import MinKProbAttack
from evaluation.attacks.min_k_plus_plus import MinKPlusPlusAttack
from evaluation.attacks.neighborhood import NeighborhoodAttack
from evaluation.attacks.recall import ReCaLLAttack
from evaluation.attacks.con_recall import ConReCaLLAttack
from evaluation.attacks.dc_pdd import DC_PDDAttack


# TODO Use decorators to link attack implementations with enum above
def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.NEIGHBOR: NeighborhoodAttack,
        AllAttacks.RECALL: ReCaLLAttack,
        AllAttacks.CON_RECALL: ConReCaLLAttack,
        AllAttacks.DC_PDD: DC_PDDAttack
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls
