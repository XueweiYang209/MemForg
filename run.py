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
import copy
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
        # 计算总的邻居操作数
        total_samples = math.ceil(n_samples / batch_size) * batch_size
        total_neighbor_operations = total_samples * len(n_perturbation_list)

    # 创建全局邻居进度条
    global_neighbor_progress = None
    if total_neighbor_operations > 0:
        global_neighbor_progress = tqdm(total=total_neighbor_operations, 
                                    desc="Neighbor attacks", 
                                    position=1)
    # neighbors = None
    # if AllAttacks.NEIGHBOR in attackers_dict.keys() and neigh_config.load_from_cache:
    #     neighbors = data[f"neighbors"]
    #     print("Loaded neighbors from cache!")

    # if neigh_config and neigh_config.dump_cache:
    #     collected_neighbors = {
    #         n_perturbation: [] for n_perturbation in n_perturbation_list
    #     }

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

            # This will be a list of integers if pretokenized
            sample_information["sample"] = sample
            # if config.pretokenized:
            #     detokenized_sample = [target_model.tokenizer.decode(s) for s in sample]
            #     sample_information["detokenized"] = detokenized_sample

            # if neigh_config and neigh_config.dump_cache:
            #     neighbors_within = {n_perturbation: [] for n_perturbation in n_perturbation_list}
            # For each substring
            for i, substr in enumerate(sample):
                # compute token probabilities for sample
                s_tk_probs, s_all_probs = target_model.get_probabilities(substr, return_all_probs=True)
                    # if not config.pretokenized
                    # else target_model.get_probabilities(
                    #     detokenized_sample[i], tokens=substr, return_all_probs=True
                    # )

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
                            # detokenized_sample=(
                            #     detokenized_sample[i]
                            #     if config.pretokenized
                            #     else None
                            # ),
                            loss=loss,
                            all_probs=s_all_probs,
                            recall_dict = recall_dict
                        )
                        sample_information[attack].append(score)


                    elif attack != AllAttacks.NEIGHBOR:
                        score = attacker.attack(
                            substr,
                            probs=s_tk_probs,
                            # detokenized_sample=(
                            #     detokenized_sample[i]
                            #     if config.pretokenized
                            #     else None
                            # ),
                            loss=loss,
                            all_probs=s_all_probs,
                        )
                        sample_information[attack].append(score)
                        
                    else:
                        # For each 'number of neighbors'
                        for n_perturbation in n_perturbation_list:
                            # Use neighbors if available
                            # if neighbors:
                            #     substr_neighbors = neighbors[n_perturbation][
                            #         batch * batch_size + idx
                            #     ][i]
                            # else:
                            substr_neighbors = attacker.get_neighbors(
                                [substr], n_perturbations=n_perturbation
                            )
                                # Collect this neighbor information if neigh_config.dump_cache is True
                                # if neigh_config.dump_cache:
                                #     neighbors_within[n_perturbation].append(
                                #         substr_neighbors
                                #     )

                            # if not neigh_config.dump_cache:
                                # Only evaluate neighborhood attack when not caching neighbors
                            score = attacker.attack(
                                substr,
                                probs=s_tk_probs,
                                # detokenized_sample=(
                                #     detokenized_sample[i]
                                #     if config.pretokenized
                                #     else None
                                # ),
                                loss=loss,
                                batch_size=4,
                                substr_neighbors=substr_neighbors,
                            )

                            sample_information[
                                f"{attack}-{n_perturbation}"
                            ].append(score)

                            # 更新全局进度条
                            if global_neighbor_progress:
                                global_neighbor_progress.update(1)

            # if neigh_config and neigh_config.dump_cache:
            #     for n_perturbation in n_perturbation_list:
            #         collected_neighbors[n_perturbation].append(
            #             neighbors_within[n_perturbation]
            #         )

            # Add the scores we collected for each sample for each
            # attack into to respective list for its classification
            results.append(sample_information)

    
    if global_neighbor_progress:
        global_neighbor_progress.close()

    # if neigh_config and neigh_config.dump_cache:
    #     # Save p_member_text and p_nonmember_text (Lists of strings) to cache
    #     # For each perturbation
    #     for n_perturbation in n_perturbation_list:
    #         ds_object.dump_neighbors(
    #             collected_neighbors[n_perturbation],
    #             train=is_train,
    #             num_neighbors=n_perturbation,
    #             model=neigh_config.model,
    #             in_place_swap=in_place_swap,
    #         )

    # if neigh_config and neigh_config.dump_cache:
    #     print(
    #         "Data dumped! Please re-run with load_from_cache set to True in neigh_config"
    #     )
    #     exit(0)

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
                    # if config.pretokenized:
                    #     s = r["detokenized"][i]
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


# def generate_data_processed(
#     base_model,
#     mask_model,
#     raw_data_member,
#     batch_size: int,
#     raw_data_non_member: List[str] = None
# ):
#     torch.manual_seed(42)
#     np.random.seed(42)
#     data = {
#         "nonmember": [],
#         "member": [],
#     }

#     seq_lens = []
#     num_batches = (len(raw_data_member) // batch_size) + 1
#     iterator = tqdm(range(num_batches), desc="Generating samples")
#     for batch in iterator:
#         member_text = raw_data_member[batch * batch_size : (batch + 1) * batch_size]
#         non_member_text = raw_data_non_member[batch * batch_size : (batch + 1) * batch_size]

#         # TODO make same len
#         for o, s in zip(non_member_text, member_text):
#             # o, s = data_utils.trim_to_shorter_length(o, s, config.max_words)

#             # # add to the data
#             # assert len(o.split(' ')) == len(s.split(' '))
#             if not config.full_doc:
#                 seq_lens.append((len(s.split(" ")), len(o.split())))

#             if config.tok_by_tok:
#                 for tok_cnt in range(len(o.split(" "))):
#                     data["nonmember"].append(" ".join(o.split(" ")[: tok_cnt + 1]))
#                     data["member"].append(" ".join(s.split(" ")[: tok_cnt + 1]))
#             else:
#                 data["nonmember"].append(o)
#                 data["member"].append(s)

#     # if config.tok_by_tok:
#     n_samples = len(data["nonmember"])
#     # else:
#     #     n_samples = config.n_samples
#     if config.pre_perturb_pct > 0:
#         print(
#             f"APPLYING {config.pre_perturb_pct}, {config.pre_perturb_span_length} PRE-PERTURBATIONS"
#         )
#         print("MOVING MASK MODEL TO GPU...", end="", flush=True)
#         mask_model.load()
#         data["member"] = mask_model.generate_neighbors(
#             data["member"],
#             config.pre_perturb_span_length,
#             config.pre_perturb_pct,
#             config.chunk_size,
#             ceil_pct=True,
#         )
#         print("MOVING BASE MODEL TO GPU...", end="", flush=True)
#         base_model.load()

#     return data, seq_lens, n_samples


def generate_data(
    dataset: str,
    # train: bool = True,
    presampled: str = None,
    # specific_source: str = None,
    mask_model_tokenizer = None
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=True,
        mask_tokenizer=mask_model_tokenizer
    )
    return data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


def main(config: ExperimentConfig):
    """
    简化版main函数，支持训练前后模型对比评估
    """
    # === 第一阶段：配置解析和环境设置 ===
    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    recall_config: ReCaLLConfig = config.recall_config
    
    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

    # === 第二阶段：简化的输出目录管理 ===
    exp_name = config.experiment_name
    base_model_name = config.model_before.replace("/", "_")
    
    # 简化目录结构：实验名/模型名
    sf = os.path.join(exp_name, base_model_name)
    SAVE_FOLDER = os.path.join(env_config.tmp_results, sf)
    new_folder = os.path.join(env_config.results, sf)
    
    # 防重复运行检查
    print(f"Results will be saved to: {new_folder}")
    if os.path.isdir(new_folder):
        print(f"Experiment folder exists, not running: {new_folder}")
        exit(0)
    
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # === 第三阶段：缓存和环境准备 ===
    cache_dir = env_config.cache_dir
    print(f"Using cache dir: {cache_dir}")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # === 第四阶段：训练前模型和攻击器初始化 ===
    print("="*60)
    print("PHASE 1: INITIALIZING BEFORE-TRAINING MODEL AND ATTACKERS")
    print("="*60)
    
    # 加载训练前模型
    print(f"Loading before-training model: {config.model_before}")
    model_before = LanguageModel(config, model_name=config.model_before)
    
    # 加载参考模型（如果需要Reference攻击）
    ref_models_before = None
    if (ref_config is not None and 
        AllAttacks.REFERENCE_BASED in config.blackbox_attacks):
        ref_models_before = {
            model: ReferenceModel(config, model) 
            for model in ref_config.models
        }
    
    # ✅ 优化：一次性创建所有攻击器
    print("Initializing attackers for before-training model...")
    attackers_dict_before = get_attackers(model_before, ref_models_before, config)
    
    # ✅ 从已创建的攻击器中获取mask_model
    mask_model = None
    if (neigh_config and AllAttacks.NEIGHBOR in config.blackbox_attacks):
        if AllAttacks.NEIGHBOR in attackers_dict_before:
            mask_model = attackers_dict_before[AllAttacks.NEIGHBOR].get_mask_model()
            print("Retrieved mask model from neighborhood attacker")

    # === 第五阶段：数据加载（一次性，带正确的tokenizer） ===
    print("="*60)
    print("PHASE 2: LOADING TRAINING DATA")
    print("="*60)
    
    print(f"Loading training dataset: {config.dataset}...")
    training_data = generate_data(
        config.dataset,
        # train=True,
        presampled=config.presampled_dataset,
        mask_model_tokenizer=mask_model.tokenizer if mask_model else None
    )
    
    # === 第六阶段：训练前模型评估 ===
    print("="*60)
    print("PHASE 3: EVALUATING WITH BEFORE-TRAINING MODEL")
    print("="*60)
    
    # 加载模型到GPU
    print("Moving before-training model to GPU...")
    model_before.load()
    
    # ReCaLL攻击特殊准备
    nonmember_prefix = None
    if AllAttacks.RECALL in config.blackbox_attacks:
        assert recall_config, "Must provide a recall_config"
        num_shots = recall_config.num_shots
        # 对于训练前模型，使用训练数据的前几个样本作为非成员前缀
        nonmember_prefix = training_data[:num_shots]
    
    # 处理数据格式
    data_before = {"member": training_data}
    n_samples = len(training_data)
    
    # 执行攻击评估（训练前模型）
    data_before_eval = {
        "records": data_before["member"],
    }
    
    print("Computing MIA scores with before-training model...")
    nonmember_preds, nonmember_samples = get_mia_scores(
        data_before_eval,
        attackers_dict_before,
        # data_obj_training,
        target_model=model_before,
        ref_models=ref_models_before,
        config=config,
        # is_train=False,  # 对训练前模型，这些数据是非成员
        n_samples=n_samples,
        nonmember_prefix=nonmember_prefix
    )
    
    # 释放训练前模型资源
    print("Unloading before-training model...")
    model_before.unload()
    if ref_models_before:
        for ref_model in ref_models_before.values():
            ref_model.unload()
    # ❌ 不要卸载mask_model，训练后模型还要用
    # if mask_model:
    #     mask_model.unload()
    torch.cuda.empty_cache()
    
    # === 第七阶段：训练后模型和攻击器初始化 ===
    print("="*60)
    print("PHASE 4: INITIALIZING AFTER-TRAINING MODEL AND ATTACKERS")
    print("="*60)
    
    # 加载训练后模型
    print(f"Loading after-training model: {config.model_after}")
    model_after = LanguageModel(config, model_name=config.model_after)
    
    # 加载参考模型（如果需要Reference攻击）
    ref_models_after = None
    if (ref_config is not None and 
        AllAttacks.REFERENCE_BASED in config.blackbox_attacks):
        ref_models_after = {
            model: ReferenceModel(config, model) 
            for model in ref_config.models
        }
    
    # ✅ 优化：一次性创建所有攻击器
    print("Initializing attackers for after-training model...")
    attackers_dict_after = get_attackers(model_after, ref_models_after, config)

    # === 第八阶段：训练后模型评估 ===
    print("="*60)
    print("PHASE 5: EVALUATING WITH AFTER-TRAINING MODEL")
    print("="*60)
    
    # 加载模型到GPU
    print("Moving after-training model to GPU...")
    model_after.load()
    
    # ReCaLL攻击特殊准备（对训练后模型）
    nonmember_prefix_after = None
    if AllAttacks.RECALL in config.blackbox_attacks:
        # 对于训练后模型，我们可能需要不同的非成员前缀
        # 这里简化处理，使用相同的前缀
        nonmember_prefix_after = nonmember_prefix
    
    # 处理数据格式
    data_after = {"member": training_data}  # 对训练后模型，这些是真正的成员
    
    # 执行攻击评估（训练后模型）
    data_after_eval = {
        "records": data_after["member"],
    }
    
    print("Computing MIA scores with after-training model...")
    member_preds, member_samples = get_mia_scores(
        data_after_eval,
        attackers_dict_after,
        # data_obj_training,
        target_model=model_after,
        ref_models=ref_models_after,
        config=config,
        # is_train=True,  # 对训练后模型，这些数据是成员
        n_samples=n_samples,
        nonmember_prefix=nonmember_prefix_after
    )
    
    # === 第九阶段：计算对比指标 ===
    print("="*60)
    print("PHASE 6: COMPUTING COMPARISON METRICS")
    print("="*60)
    
    blackbox_outputs = compute_metrics_from_scores(
        member_preds,      # 训练后模型的分数（成员）
        nonmember_preds,   # 训练前模型的分数（非成员）
        member_samples,
        nonmember_samples,
        n_samples=n_samples,
    )
    
    # === 第十阶段：保存结果 ===
    print("="*60)
    print("PHASE 7: SAVING RESULTS")
    print("="*60)
    
    # 保存原始数据
    # if not config.pretokenized:
    raw_data = {
        "training_data": training_data,
        "model_before": config.model_before,
        "model_after": config.model_after,
    }
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(raw_data, f)
    
    # 保存配置
    config.save_json(os.path.join(SAVE_FOLDER, 'config.json'), indent=4)
    
    # 保存每个攻击的结果
    outputs = []
    for attack, output in blackbox_outputs.items():
        outputs.append(output)
        with open(os.path.join(SAVE_FOLDER, f"{attack}_results.json"), "w") as f:
            json.dump(output, f)
    
    # === 第十一阶段：生成可视化 ===
    neighbor_model_name = neigh_config.model if neigh_config else None
    plot_utils.save_roc_curves(
        outputs,
        save_folder=SAVE_FOLDER,
        model_name=base_model_name,
        neighbor_model_name=neighbor_model_name,
    )
    plot_utils.save_ll_histograms(outputs, save_folder=SAVE_FOLDER)
    plot_utils.save_llr_histograms(outputs, save_folder=SAVE_FOLDER)
    
    # === 第十二阶段：移动到最终目录 ===
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)
    
    print("="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {new_folder}")
    
    # 清理资源
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
