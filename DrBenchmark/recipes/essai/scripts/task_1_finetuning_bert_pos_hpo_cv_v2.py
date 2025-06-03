#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 28/04/2023 Yanis LABRAK
# Apache 2.0

import os, glob
import shutil

import json
import uuid
import logging

from utils_hpo import parse_args

import numpy as np

import evaluate
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)


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
            "Dr-BERT/ESSAI",
            name=str(args.subset),
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
    label_list = dataset.features["pos_tags"][0].names

    #  We need to remove duplicates
    seen = set()

    def is_unique(example):
        tokens_tuple = tuple(example["tokens"])
        if tokens_tuple in seen:
            return False
        seen.add(tokens_tuple)
        return True

    dataset = dataset.filter(is_unique)

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

    def getConfig(raw_labels):
        label2id = {}
        id2label = {}

        for i, class_name in enumerate(raw_labels):
            label2id[class_name] = str(i)
            id2label[str(i)] = class_name

        return label2id, id2label

    label2id, id2label = getConfig(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_and_align_labels(examples):
        label_all_tokens = False

        if args.model_name.lower().find("flaubert") != -1:
            tokenized_inputs = []
            _labels = []

            # For sentence in batch
            for _e, _label in zip(examples["tokens"], examples["pos_tags"]):
                _local = [tokenizer("<s>")["input_ids"][1]]
                _local_labels = [-100]

                # For token in sentence
                for _i, (_t, _lb) in enumerate(zip(_e, _label)):
                    tokens_word = tokenizer(_t)["input_ids"][1:-1]
                    _local.extend(tokens_word)
                    _local_labels.extend([_lb] * len(tokens_word))

                if len(_local) > 250:
                    print(f">> {len(_local)}")

                _local = _local[0 : args.max_position_embeddings - 1]
                _local_labels = _local_labels[0 : args.max_position_embeddings - 1]

                _local.append(tokenizer("</s>")["input_ids"][1])
                _local_labels.append(-100)

                padding_left = args.max_position_embeddings - len(_local)
                if padding_left > 0:
                    _local.extend([tokenizer("<pad>")["input_ids"][1]] * padding_left)
                    _local_labels.extend([-100] * padding_left)

                tokenized_inputs.append(_local)
                _labels.append(_local_labels)

            tokenized_inputs = {
                "input_ids": tokenized_inputs,
                "labels": _labels,
            }

        else:
            tokenized_inputs = tokenizer(
                list(examples["tokens"]),
                truncation=True,
                max_length=args.max_position_embeddings,
                padding="max_length",
                is_split_into_words=True,
            )

            labels = []

            for i, label in enumerate(examples[f"pos_tags"]):
                label_ids = []
                previous_word_idx = None

                word_ids = tokenized_inputs.word_ids(batch_index=i)

                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)

                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])

                    else:
                        label_ids.append(-100)

                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels

        return tokenized_inputs

    train_tokenized_datasets = (
        dataset["train"]
        .map(tokenize_and_align_labels, batched=True)
        .shuffle(seed=42)
        .shuffle(seed=42)
        .shuffle(seed=42)
    )
    if args.fewshot != 1.0:
        train_tokenized_datasets = train_tokenized_datasets.select(
            range(int(len(train_tokenized_datasets) * args.fewshot))
        )

    validation_tokenized_datasets = dataset["validation"].map(
        tokenize_and_align_labels, batched=True
    )

    test_tokenized_datasets = dataset["test"].map(
        tokenize_and_align_labels, batched=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = (
        f"DrBenchmark-ESSAI-{str(args.subset)}-{uuid.uuid4()}_fold={args.fold}"
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
            model = AutoModelForTokenClassification.from_pretrained(
                absolute_path_to_model, num_labels=len(label_list)
            )
            model.config.label2id = label2id
            model.config.id2label = id2label
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
        model = AutoModelForTokenClassification.from_pretrained(
            absolute_path_to_model, num_labels=len(label_list)
        )
        model.config.label2id = label2id
        model.config.id2label = id2label
        model.config.update(
            {
                "attention_probs_dropout_prob": args.dropout,
                "classifier_dropout": args.dropout,
                "hidden_dropout_prob": args.dropout,
            }
        )

    metric = evaluate.load("../../../metrics/seqeval.py", experiment_id=output_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def remove_dummy_label(predictions, labels):
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return true_predictions, true_labels

    def compute_metrics(p):
        predictions, labels = p

        true_predictions, true_labels = remove_dummy_label(predictions, labels)

        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

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
            train_dataset=train_tokenized_datasets,
            eval_dataset=validation_tokenized_datasets,
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
            train_dataset=train_tokenized_datasets,
            eval_dataset=validation_tokenized_datasets,
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
    trainer.model = AutoModelForTokenClassification.from_pretrained(
        f"{args.output_dir}/{output_name}_best_model"
    ).to(trainer.args.device)

    logging.info("***** Starting Evaluation *****")
    predictions, labels, _ = trainer.predict(test_tokenized_datasets)

    _true_predictions, _true_labels = remove_dummy_label(predictions, labels)

    cr_metric = metric.compute(predictions=_true_predictions, references=_true_labels)
    print(cr_metric)

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open(f"../runs/{output_name}_hpo.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": f"{args.output_dir}/{output_name}_best_model",
                "metrics": cr_metric,
                "hpo_settings": vars(args),
                "best_hp_trial": best_hp_trial
                if not do_hpo
                else best_trial.hyperparameters,
                "predictions": {
                    "identifiers": test_tokenized_datasets["id"],
                    "real_labels": _true_labels,
                    "system_predictions": _true_predictions,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
            default=np_encoder,
        )


if __name__ == "__main__":
    main()
