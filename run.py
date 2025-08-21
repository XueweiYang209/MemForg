"""
    Main entry point for running experiments with evaluation
"""
import numpy as np
import torch
from tqdm import tqdm
import datetime
import os
import json
import math
from collections import defaultdict
from typing import List, Dict

from simple_parsing import ArgumentParser
from pathlib import Path

from evaluation.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    ReCaLLConfig
)
import evaluation.data_utils as data_utils
import evaluation.plot_utils as plot_utils
from evaluation.utils import fix_seed
from evaluation.models import LanguageModel, ReferenceModel
from evaluation.attacks.all_attacks import AllAttacks, Attack
from evaluation.attacks.utils import get_attacker
from evaluation.attacks.attack_utils import (
    get_roc_metrics,
    get_precision_recall_metrics,
    get_auc_from_thresholds,
)


def get_attackers(
    target_model,
    ref_models,
    config: ExperimentConfig,
):
    # Look at all attacks, and attacks that we have implemented
    attacks = config.blackbox_attacks
    implemented_blackbox_attacks = [a.value for a in AllAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            continue
        runnable_attacks.append(a)
    attacks = runnable_attacks

    # Initialize attackers
    attackers = {}
    for attack in attacks:
        if attack != AllAttacks.REFERENCE_BASED:
            attackers[attack] = get_attacker(attack)(config, target_model)

    # Initialize reference-based attackers if specified
    if ref_models is not None:
        for name, ref_model in ref_models.items():
            attacker = get_attacker(AllAttacks.REFERENCE_BASED)(
                config, target_model, ref_model
            )
            attackers[f"{AllAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"] = attacker
    return attackers


def get_mia_scores(
    data,
    attackers_dict: Dict[str, Attack],
    # ds_object,
    target_model: LanguageModel,
    ref_models: Dict[str, ReferenceModel],
    config: ExperimentConfig,
    # is_train: bool,
    n_samples: int = None,
    batch_size: int = 50,
    **kwargs
):
    # Fix randomness
    fix_seed(config.random_seed)

    n_samples = len(data["records"]) if n_samples is None else n_samples

    # Look at all attacks, and attacks that we have implemented
    neigh_config = config.neighborhood_config

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        in_place_swap = neigh_config.original_tokenization_swap

    results = []

    # Count total number of neighborhood operations
    total_neighbor_operations = 0
    if AllAttacks.NEIGHBOR in attackers_dict:
        # Calculate total neighbor operations
        total_samples = math.ceil(n_samples / batch_size) * batch_size
        total_neighbor_operations = total_samples * len(n_perturbation_list)

    # Create global neighbor progress bar
    global_neighbor_progress = None
    if total_neighbor_operations > 0:
        global_neighbor_progress = tqdm(total=total_neighbor_operations, 
                                    desc="Neighbor attacks", 
                                    position=1)

    recall_config = config.recall_config
    if recall_config:
        nonmember_prefix = kwargs.get("nonmember_prefix", None)
        num_shots = recall_config.num_shots
        avg_length = int(np.mean([len(target_model.tokenizer.encode(ex)) for ex in data["records"]]))
        recall_dict = {"prefix":nonmember_prefix, "num_shots":num_shots, "avg_length":avg_length}

    # For each batch of data
    # TODO: Batch-size isn't really "batching" data - change later
    for batch in tqdm(range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs]
                if config.full_doc
                else [texts[idx]]
            )

            sample_information["sample"] = sample
            # For each substring
            for i, substr in enumerate(sample):
                # compute token probabilities for sample
                s_tk_probs, s_all_probs = target_model.get_probabilities(substr, return_all_probs=True)
                # Always compute LOSS score. Also helpful for reference-based and many other attacks.
                loss = target_model.get_ll(substr, probs=s_tk_probs)
                sample_information[AllAttacks.LOSS].append(loss)

                # TODO: Shift functionality into each attack entirely, so that this is just a for loop
                # For each attack
                for attack, attacker in attackers_dict.items():
                    # LOSS already added above, Reference handled later
                    if attack.startswith(AllAttacks.REFERENCE_BASED) or attack == AllAttacks.LOSS:
                        continue

                    if attack == AllAttacks.RECALL:
                        score = attacker.attack(
                            substr,
                            probs = s_tk_probs,
                            loss=loss,
                            all_probs=s_all_probs,
                            recall_dict = recall_dict
                        )
                        sample_information[attack].append(score)


                    elif attack != AllAttacks.NEIGHBOR:
                        score = attacker.attack(
                            substr,
                            probs=s_tk_probs,
                            loss=loss,
                            all_probs=s_all_probs,
                        )
                        sample_information[attack].append(score)
                        
                    else:
                        # For each 'number of neighbors'
                        for n_perturbation in n_perturbation_list:
                            substr_neighbors = attacker.get_neighbors(
                                [substr], n_perturbations=n_perturbation
                            )
                            score = attacker.attack(
                                substr,
                                probs=s_tk_probs,
                                loss=loss,
                                batch_size=4,
                                substr_neighbors=substr_neighbors,
                            )

                            sample_information[
                                f"{attack}-{n_perturbation}"
                            ].append(score)

                            # Update global progress bar
                            if global_neighbor_progress:
                                global_neighbor_progress.update(1)

            # Add the scores we collected for each sample for each
            # attack into to respective list for its classification
            results.append(sample_information)

    
    if global_neighbor_progress:
        global_neighbor_progress.close()

    # Perform reference-based attacks
    if ref_models is not None:
        for name, ref_model in ref_models.items():
            ref_key = f"{AllAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"
            attacker = attackers_dict.get(ref_key, None)
            if attacker is None:
                continue

            # Update collected scores for each sample with ref-based attack scores
            for r in tqdm(results, desc="Ref scores"):
                ref_model_scores = []
                for i, s in enumerate(r["sample"]):
                    score = attacker.attack(s, probs=None,
                                                loss=r[AllAttacks.LOSS][i])
                    ref_model_scores.append(score)
                r[ref_key].extend(ref_model_scores)

            attacker.unload()
    else:
        print("No reference models specified, skipping Reference-based attacks")

    # Rearrange the nesting of the results dict and calculated aggregated score for sample
    # attack -> member/nonmember -> list of scores
    samples = []
    predictions = defaultdict(lambda: [])
    for r in results:
        samples.append(r["sample"])
        for attack, scores in r.items():
            if attack != "sample" and attack != "detokenized":
                # TODO: Is there a reason for the np.min here?
                predictions[attack].append(np.min(scores))

    return predictions, samples


def compute_metrics_from_scores(
        preds_member: dict,
        preds_nonmember: dict,
        samples_member: List,
        samples_nonmember: List,
        n_samples: int):

    attack_keys = list(preds_member.keys())
    if attack_keys != list(preds_nonmember.keys()):
        raise ValueError("Mismatched attack keys for member/nonmember predictions")

    # Collect outputs for each attack
    blackbox_attack_outputs = {}
    for attack in attack_keys:
        preds_member_ = preds_member[attack]
        preds_nonmember_ = preds_nonmember[attack]

        fpr, tpr, roc_auc, roc_auc_res, thresholds = get_roc_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_,
            perform_bootstrap=True,
            return_thresholds=True,
        )
        tpr_at_low_fpr = {
            upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
            for upper_bound in config.fpr_list
        }
        p, r, pr_auc = get_precision_recall_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_
        )

        print(
            f"{attack}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, tpr_at_low_fpr: {tpr_at_low_fpr}"
        )
        blackbox_attack_outputs[attack] = {
            "name": f"{attack}_threshold",
            "predictions": {
                "member": preds_member_,
                "nonmember": preds_nonmember_,
            },
            "info": {
                "n_samples": n_samples,
            },
            "raw_results": {
                "member": samples_member,
                "nonmember": samples_nonmember,
            },
            "metrics": {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
                "bootstrap_roc_auc_std": roc_auc_res.standard_error,
                "tpr_at_low_fpr": tpr_at_low_fpr,
                "thresholds": thresholds,
            },
            "pr_metrics": {
                "pr_auc": pr_auc,
                "precision": p,
                "recall": r,
            },
            "loss": 1 - pr_auc,
        }

    return blackbox_attack_outputs


def generate_data(
    dataset: str,
    presampled: str = None,
    mask_model_tokenizer = None
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(mask_tokenizer=mask_model_tokenizer)
    return data


def main(config: ExperimentConfig):
    """
    main function for before/after-training model comparison evaluation
    """
    # === Config parsing and environment setup ===
    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    recall_config: ReCaLLConfig = config.recall_config

    # === Output directory management ===
    exp_name = config.experiment_name
    base_model_name = config.model_before.replace("/", "_")
    
    sf = os.path.join(exp_name, base_model_name)
    SAVE_FOLDER = os.path.join(env_config.tmp_results, sf)
    new_folder = os.path.join(env_config.results, sf)
    
    # Prevent duplicate runs check
    print(f"Results will be saved to: {new_folder}")
    if os.path.isdir(new_folder):
        print(f"Experiment folder exists, not running: {new_folder}")
        exit(0)
    
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # === Cache and environment preparation ===
    cache_dir = env_config.cache_dir
    print(f"Using cache dir: {cache_dir}")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # === Before-training model and attacker initialization ===
    print("="*60)
    print("PHASE 1: INITIALIZING BEFORE-TRAINING MODEL AND ATTACKERS")
    print("="*60)
    
    # Load before-training model
    print(f"Loading before-training model: {config.model_before}")
    model_before = LanguageModel(config, model_name=config.model_before)
    
    # Load reference models (if Reference attack is needed)
    ref_models_before = None
    if (ref_config is not None and 
        AllAttacks.REFERENCE_BASED in config.blackbox_attacks):
        ref_models_before = {
            model: ReferenceModel(config, model) 
            for model in ref_config.models
        }
    
    print("Initializing attackers for before-training model...")
    attackers_dict_before = get_attackers(model_before, ref_models_before, config)
    
    # Get mask_model from created attackers
    mask_model = None
    if (neigh_config and AllAttacks.NEIGHBOR in config.blackbox_attacks):
        if AllAttacks.NEIGHBOR in attackers_dict_before:
            mask_model = attackers_dict_before[AllAttacks.NEIGHBOR].get_mask_model()
            print("Retrieved mask model from neighborhood attacker")

    # === Data loading (one-time, with correct tokenizer) ===
    print("="*60)
    print("PHASE 2: LOADING TRAINING DATA")
    print("="*60)
    
    print(f"Loading training dataset: {config.dataset}...")
    training_data = generate_data(
        config.dataset,
        presampled=config.presampled_dataset,
        mask_model_tokenizer=mask_model.tokenizer if mask_model else None
    )
    
    # === Before-training model evaluation ===
    print("="*60)
    print("PHASE 3: EVALUATING WITH BEFORE-TRAINING MODEL")
    print("="*60)
    
    # Load model to GPU
    print("Moving before-training model to GPU...")
    model_before.load()
    
    # ReCaLL attack special preparation
    nonmember_prefix = None
    if AllAttacks.RECALL in config.blackbox_attacks:
        assert recall_config, "Must provide a recall_config"
        num_shots = recall_config.num_shots
        # TODO 对于训练前模型，使用训练数据的前几个样本作为非成员前缀
        nonmember_prefix = training_data[:num_shots]
    
    # Process data format
    data_before = {"member": training_data}
    n_samples = len(training_data)
    
    # Execute attack evaluation (before-training model)
    data_before_eval = {
        "records": data_before["member"],
    }
    
    print("Computing MIA scores with before-training model...")
    nonmember_preds, nonmember_samples = get_mia_scores(
        data_before_eval,
        attackers_dict_before,
        target_model=model_before,
        ref_models=ref_models_before,
        config=config,
        n_samples=n_samples,
        nonmember_prefix=nonmember_prefix
    )
    
    # Release before-training model resources
    print("Unloading before-training model...")
    model_before.unload()
    if ref_models_before:
        for ref_model in ref_models_before.values():
            ref_model.unload()
    # Don't unload mask_model, after-training model still needs it
    # if mask_model:
    #     mask_model.unload()
    torch.cuda.empty_cache()
    
    # === After-training model and attacker initialization ===
    print("="*60)
    print("PHASE 4: INITIALIZING AFTER-TRAINING MODEL AND ATTACKERS")
    print("="*60)
    
    # Load after-training model
    print(f"Loading after-training model: {config.model_after}")
    model_after = LanguageModel(config, model_name=config.model_after)
    
    # Load reference models (if Reference attack is needed)
    ref_models_after = None
    if (ref_config is not None and 
        AllAttacks.REFERENCE_BASED in config.blackbox_attacks):
        ref_models_after = {
            model: ReferenceModel(config, model) 
            for model in ref_config.models
        }
    
    print("Initializing attackers for after-training model...")
    attackers_dict_after = get_attackers(model_after, ref_models_after, config)

    # === After-training model evaluation ===
    print("="*60)
    print("PHASE 5: EVALUATING WITH AFTER-TRAINING MODEL")
    print("="*60)
    
    # Load model to GPU
    print("Moving after-training model to GPU...")
    model_after.load()
    
    # ReCaLL attack special preparation (for after-training model)
    nonmember_prefix_after = None
    if AllAttacks.RECALL in config.blackbox_attacks:
        # TODO 对于训练后模型，我们可能需要不同的非成员前缀
        # 这里简化处理，使用相同的前缀
        nonmember_prefix_after = nonmember_prefix
    
    # Process data format
    data_after = {"member": training_data}  # For after-training model, these are real members
    
    # Execute attack evaluation (after-training model)
    data_after_eval = {
        "records": data_after["member"],
    }
    
    print("Computing MIA scores with after-training model...")
    member_preds, member_samples = get_mia_scores(
        data_after_eval,
        attackers_dict_after,
        target_model=model_after,
        ref_models=ref_models_after,
        config=config,
        n_samples=n_samples,
        nonmember_prefix=nonmember_prefix_after
    )
    
    # === Compute comparison metrics ===
    print("="*60)
    print("PHASE 6: COMPUTING COMPARISON METRICS")
    print("="*60)
    
    blackbox_outputs = compute_metrics_from_scores(
        member_preds,      # After-training model scores (members)
        nonmember_preds,   # Before-training model scores (non-members)
        member_samples,
        nonmember_samples,
        n_samples=n_samples,
    )
    
    # === Save results ===
    print("="*60)
    print("PHASE 7: SAVING RESULTS")
    print("="*60)
    
    # Save raw data
    raw_data = {
        "training_data": training_data,
        "model_before": config.model_before,
        "model_after": config.model_after,
    }
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(raw_data, f)
    
    # Save config
    config.save_json(os.path.join(SAVE_FOLDER, 'config.json'), indent=4)
    
    # Save results for each attack
    outputs = []
    for attack, output in blackbox_outputs.items():
        outputs.append(output)
        with open(os.path.join(SAVE_FOLDER, f"{attack}_results.json"), "w") as f:
            json.dump(output, f)
    
    # === Generate visualizations ===
    neighbor_model_name = neigh_config.model if neigh_config else None
    plot_utils.save_roc_curves(
        outputs,
        save_folder=SAVE_FOLDER,
        model_name=base_model_name,
        neighbor_model_name=neighbor_model_name,
    )
    plot_utils.save_ll_histograms(outputs, save_folder=SAVE_FOLDER)
    plot_utils.save_llr_histograms(outputs, save_folder=SAVE_FOLDER)
    
    # === Move to final directory ===
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)
    
    print("="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {new_folder}")
    
    model_after.unload()
    if ref_models_after:
        for ref_model in ref_models_after.values():
            ref_model.unload()
    if mask_model:
        mask_model.unload()


if __name__ == "__main__":
    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=True)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    # Fix randomness
    fix_seed(config.random_seed)
    # Call main function
    main(config)
