import os
import argparse
from argparse import Namespace

import yaml
import torch

from transformers import TrainingArguments


class TrainingArgumentsWithMPSSupport(TrainingArguments):

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Default YAML configuration file")
    parser.add_argument("--model_name", type=str, required=False, help="HuggingFace Hub model name")
    parser.add_argument("--output_dir", type=str, required=False, help="Path were the model will be saved")
    parser.add_argument("--data_dir", type=str, required=False, help="Path where the data are stored")
    parser.add_argument("--per_device_train_batch_size", type=int, required=False, help="Training batch size")
    parser.add_argument("--max_position_embeddings", type=int, required=False, help="Max position embeddings")
    parser.add_argument("--weight_decay",            type=str, required=False, help="Weight decay")
    parser.add_argument("--learning_rate",           type=str, required=False, help="Learning rate")
    parser.add_argument("--subset", type=str, required=False, help="Corpus subset")
    parser.add_argument("--fewshot", type=float, required=False,
                        help="Percentage of the train subset used during training", default=1.0)
    parser.add_argument("--offline", type=bool, required=False, help="Use local huggingface dataset", default=False)
    parser.add_argument("--metrics", type=bool, required=False, help="Main metric for the task ")
    parser.add_argument("--direction", type=bool, required=False, help="Direction the metric needs to be optimized")
    parser.add_argument("--num_train_epochs",                  type=str,   required=False, help="Training epochs")
    parser.add_argument("--gradient_accumulation_steps",                  type=str,   required=False, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio",                  type=str,   required=False, help="Warmup ratio")
    parser.add_argument("--dropout",                  type=str,   required=False, help="Dropout")
    parser.add_argument("--reduction_factor",                  type=int,   required=False, help="Reduction factor")
    parser.add_argument("--grace_period",                  type=int,   required=False, help="Grace period")
    parser.add_argument("--max_t",                  type=int,   required=False, help="Max of iteration before stopping a trial")
    parser.add_argument("--n_trials",                  type=int,   required=False, help="Number of trials")
    parser.add_argument("--fold",                  type=int,   required=False, help="Number of trials", default=1)


    args = parser.parse_args()
    args = vars(args)

    overall_args_yaml = yaml.load(open("../../../config.yaml"), Loader=yaml.FullLoader)
    args["offline"] = overall_args_yaml["offline"]

    args_yaml = yaml.load(open(args["config"]), Loader=yaml.FullLoader)

    for k in args.keys():

        if args[k] == None:
            args[k] = args_yaml[k]

    args["output_dir"] = args["output_dir"].rstrip('/')

    if args["offline"] == True:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        model_name_clean = args['model_name'].lower().replace('/', '_')
        args["model_name"] = f"../../../models/{model_name_clean}"

    print(f">> Model path: >>{args['model_name']}<<")

    return Namespace(**args)
