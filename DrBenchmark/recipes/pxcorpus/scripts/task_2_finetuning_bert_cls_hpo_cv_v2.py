#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     To validate 12/06/2023 Yanis LABRAK
# Apache 2.0

import os, glob
import shutil

import uuid
import json
import logging

from utils_hpo_cls import parse_args

import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    accuracy_score,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


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
            "Dr-BERT/PxCorpus",
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
    labels_list = dataset.features["label"].names

    #  We will remove the duplicates and the sequences with no labels
    seen = set()

    def duplicate_filter(example):
        if tuple(example["tokens"]) in seen:
            return False
        seen.add(tuple(example["tokens"]))
        return True

    dataset = dataset.filter(duplicate_filter)

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
        text = f"{' '.join(e['tokens'])}"

        res = tokenizer(
            text,
            truncation=True,
            max_length=args.max_position_embeddings,
            padding="max_length",
        )
        res["text"] = text

        res["label"] = e["label"]

        return res

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
    dataset_train = dataset_train.remove_columns(["text"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    dataset_val = dataset_val.remove_columns(["text"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    dataset_test = dataset_test.remove_columns(["text"])
    dataset_test.set_format("torch")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-PxCorpus-cls-{str(uuid.uuid4().hex)}_fold={args.fold}"

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
            "gradient_accumulation_steps",
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
                absolute_path_to_model, num_labels=len(labels_list)
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
                "gradient_accumulation_steps": eval(args.gradient_accumulation_steps),
                "weight_decay": eval(args.weight_decay),
                "warmup_ratio": eval(args.warmup_ratio),
                "dropout": eval(args.dropout),
            }
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            absolute_path_to_model, num_labels=len(labels_list)
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
                trials_current_best = max(
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
                            > trial.metric_analysis["eval_" + args.metrics][
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
        best_result = max(
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
    predictions, labels, _ = trainer.predict(dataset_test)
    predictions = np.argmax(predictions, axis=1)

    labels = [labels_list[l] for l in labels.tolist()]
    predictions = [labels_list[p] for p in predictions.tolist()]

    f1_score = classification_report(
        labels,
        predictions,
        digits=4,
    )
    print(f1_score)

    with open(f"../runs/{output_name}_hpo.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": f"{args.output_dir}/{output_name}_best_model",
                "metrics": classification_report(labels, predictions, output_dict=True),
                "hpo_settings": vars(args),
                "best_hp_trial": best_hp_trial
                if not do_hpo
                else best_trial.hyperparameters,
                "predictions": {
                    "identifiers": dataset["test"]["id"],
                    "real_labels": labels,
                    "system_predictions": predictions,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()
