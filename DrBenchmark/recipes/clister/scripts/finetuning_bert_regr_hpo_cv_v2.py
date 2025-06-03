#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 29/04/2023 Yanis LABRAK
# Apache 2.0

import os, glob
import shutil

import uuid
import json
import logging

from utils_hpo import parse_args

from dataclasses import dataclass
import numpy as np
from scipy import stats
from datasets import load_dataset, load_from_disk, concatenate_datasets

from sklearn.metrics import mean_squared_error

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def EDRM(ref, systm, minVal=0, maxVal=5, boundAdjustment=True):
    n = len(ref)
    total_distance = 0

    for id in ref.keys():
        r = ref[id]

        if id not in systm:
            raise ValueError(f"System did not provide a score for the id {id}")
        elif boundAdjustment:
            # As the task delimits a range score from 0 to 5, every system should bound its output to this range
            h = max(min(systm[id], maxVal), minVal)
        else:
            h = systm[id]

        # In the function, dmax computes the highest distance from the reference score (not the prediction)
        max_distance = max(abs(minVal - r), abs(maxVal - r))
        total_distance += 1 - abs(r - h) / max_distance

    return total_distance / n


def RMSE(ref, systm):
    r = [v for k, v in sorted(ref.items())]
    s = [v for k, v in sorted(systm.items())]

    return mean_squared_error(r, s, squared=False)


def SpMnCorr(ref, systm, alpha=0.05):
    r = [v for k, v in sorted(ref.items())]
    s = [v for k, v in sorted(systm.items())]

    if len(r) == len(s):
        c, p = stats.spearmanr(r, s)
        if p > alpha:
            print(
                "Spearman Correlation: reference and system result are not correlated"
            )
        else:
            print("Spearman Correlation: reference and system result are correlated")
        return [c, p]
    else:
        return ["error", "error"]


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if args.offline == True:
        dataset = load_from_disk(f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/")
    else:
        dataset = load_dataset(
            "Dr-BERT/CLISTER",
            name="source",
            data_dir=args.data_dir,
        )
    args.fold -= 1
    # Retrieve past best_hp_trial, if any:
    # Define the search pattern
    search_pattern = "../runs/*_fold*.json"
    do_hpo = True

    # Use glob to find matching files
    matching_files = glob.glob(search_pattern)

    for file in matching_files:
        with open(file, "r", encoding="utf-8") as f:
            data_fold = json.load(f)

        model = data_fold["hpo_settings"]["model_name"].split("/")[-1]
        if model != args.model_name.split("/")[-1]:
            continue
        if data_fold["hpo_settings"]["fold"] != args.fold:
            continue

        best_hp_trial = data_fold["best_hp_trial"]
        for key, value in best_hp_trial.items():
            setattr(args, key, value)
        do_hpo = False

    # Concatenate all the splits
    dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )

    #  Shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # Create 5 shards (folds)
    num_folds = 5
    shards = [dataset.shard(num_shards=num_folds, index=i) for i in range(num_folds)]

    # Allocate each shard to a split
    dataset = {
        "test": shards[args.fold],
        "validation": shards[(args.fold + 1) % num_folds],
        "train": concatenate_datasets(
            [
                shards[(args.fold + 2) % num_folds],
                shards[(args.fold + 3) % num_folds],
                shards[(args.fold + 4) % num_folds],
            ]
        ),
    }
    # In order to get 10% of  validation set, we allocate half of the validation set to the training set
    dataset["train"] = concatenate_datasets(
        [dataset["validation"].shard(num_shards=2, index=0), dataset["train"]]
    )
    dataset["validation"] = dataset["validation"].shard(num_shards=2, index=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess_function(e):
        res = {
            "input_ids": {
                "cls_token_id": tokenizer.cls_token_id,
                "sep_token_id": tokenizer.sep_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "text_1_ids": tokenizer.encode(
                    e["text_1"], truncation=True, add_special_tokens=False
                ),
                "text_2_ids": tokenizer.encode(
                    e["text_2"], truncation=True, add_special_tokens=False
                ),
            },
            "label": float(e["label"]),
        }
        return res

    @dataclass
    class collate_fn:
        tokenizer: PreTrainedTokenizerBase
        padding = "max_length"

        def __call__(self, features):
            batch = []
            for i in range(len(features)):
                if (
                    len(features[i]["input_ids"]["text_1_ids"])
                    + len(features[i]["input_ids"]["text_2_ids"])
                    + 3
                    > args.max_position_embeddings
                ):
                    features[i]["input_ids"]["text_1_ids"] = features[i]["input_ids"][
                        "text_1_ids"
                    ][: args.max_position_embeddings // 2 - 2]
                    features[i]["input_ids"]["text_2_ids"] = features[i]["input_ids"][
                        "text_2_ids"
                    ][: args.max_position_embeddings // 2 - 1]
                if np.random.rand() < 0.5:
                    token_ids = (
                        [tokenizer.cls_token_id]
                        + features[i]["input_ids"]["text_1_ids"]
                        + [tokenizer.sep_token_id]
                        + features[i]["input_ids"]["text_2_ids"]
                        + [tokenizer.eos_token_id]
                    )
                else:
                    token_ids = (
                        [tokenizer.cls_token_id]
                        + features[i]["input_ids"]["text_2_ids"]
                        + [tokenizer.sep_token_id]
                        + features[i]["input_ids"]["text_1_ids"]
                        + [tokenizer.eos_token_id]
                    )

                batch.append(
                    {
                        "input_ids": token_ids,
                        "attention_mask": [1 for _ in token_ids],
                        "labels": features[i]["label"],
                    }
                )

            batch = self.tokenizer.pad(
                batch,
                padding=self.padding,
                max_length=args.max_position_embeddings,
                return_tensors="pt",
            )
            return batch

    data_collator = collate_fn(tokenizer=tokenizer)
    dataset_train = (
        dataset["train"]
        .map(preprocess_function, batched=False)
        .shuffle(seed=42)
        .shuffle(seed=42)
        .shuffle(seed=42)
    )
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(
            range(int(len(dataset_train) * args.fewshot))
        )

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    dataset_test_ids = list(dataset["test"]["id"])

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = (
        f"DrBenchmark-CLISTER-regression-{str(uuid.uuid4().hex)}_fold={args.fold}"
    )

    training_args = {
        k: v
        for k, v in vars(args).items()
        if k
        in [
            "per_device_train_batch_size",
            "learning_rate",
            "num_train_epochs",
            "weight_decay",
            "warmup_ratio",
        ]
    }
    training_args_base = {
        "output_dir": f"{args.output_dir}/{output_name}",
        "evaluation_strategy": "steps",
        "eval_steps": 0.1,
        "save_strategy": "steps",
        "save_steps": 0.1,
        "bf16": True,
        "push_to_hub": False,
        "metric_for_best_model": args.metrics,
        "greater_is_better": True if args.direction[0] == "max" else False,
    }
    training_args = {**training_args_base, **training_args}
    training_args = TrainingArguments(
        **training_args_base if do_hpo else training_args,
    )

    absolute_path_to_model = os.path.abspath(args.model_name)
    if do_hpo:

        def model_init(trial):
            model = AutoModelForSequenceClassification.from_pretrained(
                absolute_path_to_model, num_labels=1
            )
            if trial is not None:
                model.config.update(
                    {
                        "attention_probs_dropout_prob": trial["dropout"],
                        "classifier_dropout": trial["dropout"],
                        "hidden_dropout_prob": trial["dropout"],
                    }
                )
                print(model.config)
            return model

        def hp_space(trial):
            from ray import tune

            return {
                "learning_rate": eval(args.learning_rate),
                "num_train_epochs": eval(args.num_train_epochs),
                "weight_decay": eval(args.weight_decay),
                "warmup_ratio": eval(args.warmup_ratio),
                "dropout": eval(args.dropout),
            }
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            absolute_path_to_model, num_labels=1
        )
        model.config.update(
            {
                "attention_probs_dropout_prob": args.dropout,
                "classifier_dropout": args.dropout,
                "hidden_dropout_prob": args.dropout,
            }
        )

    from transformers import TrainerCallback, TrainerControl, TrainerState

    class SaveAndEvaluateLastStepCallback(TrainerCallback):
        """A custom callback to save and evaluate the model at the last training step."""

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            # Check if we're at the last step of training
            if state.global_step == state.max_steps:
                # Trigger evaluation
                control.should_evaluate = True
                # Trigger saving
                control.should_save = True

    if do_hpo:
        trainer = Trainer(
            args=training_args,
            model_init=model_init,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[SaveAndEvaluateLastStepCallback],
        )

        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune.schedulers import ASHAScheduler
        from ray.train import CheckpointConfig

        search_alg = HyperOptSearch(
            metric="eval_" + args.metrics, mode=args.direction[0]
        )
        scheduler = ASHAScheduler(
            metric="eval_" + args.metrics,
            mode=args.direction[0],
            reduction_factor=args.reduction_factor,
            grace_period=args.grace_period,
            max_t=args.max_t,
        )

        from ray import tune

        class CleanupCallback(tune.Callback):
            def on_trial_complete(self, iteration, trials, trial, **info):
                trials_current_best = min(
                    [
                        trial.metric_analysis["eval_" + args.metrics][args.direction[0]]
                        for trial in trials
                        if "eval_" + args.metrics in trial.metric_analysis.keys()
                    ]
                )
                for trial in trials:
                    if trial.status == "TERMINATED":
                        if (
                            trials_current_best
                            < trial.metric_analysis["eval_" + args.metrics][
                                args.direction[0]
                            ]
                        ):
                            self.cleanup_trial(trial)
                        else:
                            # Make sure it did all the iterations:
                            print(
                                f"Current best: {trial.trial_id} with eval_{args.metrics}: {trial.metric_analysis['eval_' + args.metrics][args.direction[0]]} at iteration {trial.last_result['training_iteration']}"
                            )

            def cleanup_trial(self, trial):
                # clearning up all the /tmp models saved for this trial
                print(f"Cleaning up trial {trial.trial_id}")
                if os.path.exists(trial.path):
                    checkpoint_dir = [
                        d for d in os.listdir(trial.path) if "checkpoint" in d
                    ]
                    for d in checkpoint_dir:
                        shutil.rmtree(trial.path + "/" + d)

                tmp_models = (
                    "/".join(trial.local_experiment_path.split("/")[:-1])
                    + "/working_dirs/save_models/"
                )
                tmp_models += os.listdir(tmp_models)[0] + "/run-" + str(trial.trial_id)
                if os.path.exists(tmp_models):
                    shutil.rmtree(tmp_models)

        best_trial = trainer.hyperparameter_search(
            direction=args.direction[1],
            backend="ray",
            search_alg=search_alg,
            scheduler=scheduler,
            hp_space=hp_space,
            resources_per_trial={"cpu": 4, "gpu": 1},
            n_trials=args.n_trials,
            checkpoint_config=CheckpointConfig(
                checkpoint_score_attribute="eval_" + args.metrics,
                num_to_keep=1,
                checkpoint_score_order=args.direction[0],
            ),
            callbacks=[CleanupCallback()],
        )

        best_trial_number = best_trial.run_summary.get_best_trial(
            mode=args.direction[0], metric="eval_" + args.metrics, scope="all"
        )
        best_result = min(
            best_trial.run_summary.trial_dataframes[best_trial_number.trial_id][
                "eval_" + args.metrics
            ]
        )
        best_result_id = (
            best_trial.run_summary.trial_dataframes[best_trial_number.trial_id][
                "eval_" + args.metrics
            ]
            == best_result
        )
        best_iteration = best_trial.run_summary.trial_dataframes[
            best_trial_number.trial_id
        ]["training_iteration"][best_result_id].values[0]
        best_checkpoint = best_trial.run_summary.get_best_checkpoint(
            best_trial_number, mode=args.direction[0], metric="eval_" + args.metrics
        )

        logging.info("***** Save the best model *****")
        best_checkpoint_path = glob.glob(
            best_checkpoint.path + "/checkpoint-*", recursive=True
        )[0]
        shutil.copytree(
            best_checkpoint_path, f"{args.output_dir}/{output_name}_best_model"
        )
        shutil.rmtree(f"{args.output_dir}/{output_name}")
        shutil.rmtree("/".join(best_checkpoint.path.split("/")[:-2]))
        shutil.rmtree("/tmp/ray/")
        print(
            f"Current best: {best_trial_number.trial_id} with eval_{args.metrics}: {best_result} at iteration {best_iteration}"
        )
    else:
        trainer = Trainer(
            args=training_args,
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[SaveAndEvaluateLastStepCallback],
        )
        trainer.train()

        #     save the model
        trainer.save_model(f"{args.output_dir}/{output_name}_best_model")
        shutil.rmtree(f"{args.output_dir}/{output_name}")

    logging.info("***** Load the best model *****")
    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        f"{args.output_dir}/{output_name}_best_model"
    ).to(trainer.args.device)

    logging.info("***** Starting Evaluation *****")
    _predictions, _labels, _ = trainer.predict(dataset_test)
    predictions = {id: p for id, p in zip(dataset_test_ids, _predictions)}
    labels = {id: p for id, p in zip(dataset_test_ids, _labels)}

    edrm = EDRM(labels, predictions)
    print(">> EDRM: ", edrm)

    coeff, p = SpMnCorr(labels, predictions)
    print(">> Spearman Correlation: ", coeff, "(", p, ")")

    rmse = mean_squared_error(_labels, _predictions, squared=False)
    print(">> RMSE: ", rmse)

    with open(f"../runs/{output_name}_hpo.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": f"{args.output_dir}/{output_name}_best_model",
                "metrics": {
                    "EDRM": float(edrm),
                    "RMSE": float(rmse),
                    "spearman_correlation_coef": float(coeff),
                    "spearman_correlation_p": float(p),
                },
                "hpo_settings": vars(args),
                "best_hp_trial": best_hp_trial
                if not do_hpo
                else best_trial.hyperparameters,
                "predictions": {
                    "identifiers": dataset_test_ids,
                    "real_labels": _labels.tolist(),
                    "system_predictions": [float(p[0]) for p in _predictions.tolist()],
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()
