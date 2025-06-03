import numpy as np
from shutil import copyfile
from typing import List, Optional, Tuple
from transformers.utils import (
    logging,
)
logging.set_verbosity_info()
from transformers.trainer_utils import get_last_checkpoint
logger = logging.get_logger(__name__)
import sentencepiece as spm

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

SPIECE_UNDERLINE = "▁"
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-roberta-base": 512,
}

class SPMTokenizer(PreTrainedTokenizer):

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab_files_names = {"vocab_file": vocab_file}
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:


        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:


        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        pass
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

import os, sys
import random
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from itertools import cycle


class LineByLineTextDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):

        # self.process_number = process_number
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.example = []
        with open(self.file_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if (len(line) > 0 and not line.isspace()):
                    # self.example.append(int(line))
                    self.example.append(line.rstrip())
            self.example = self.tokenizer(self.example, add_special_tokens=True, truncation=False)['input_ids']
            self.example = [item for sublist in self.example for item in sublist]
            self.example = [self.example[i:i + self.block_size] for i in range(0, len(self.example), self.block_size)]
            if len(self.example[-1]) != self.block_size:
                self.example = self.example[:-1]
            self.example = [{"input_ids": torch.tensor(example, dtype=torch.long)} for example in self.example]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        return self.example[i]

class LineByLineTextStreamer(IterableDataset):
    def __init__(self, tokenizer, file_path: str, block_size: int, process_number: int):
        self.process_number = process_number
        self.file_path = file_path + ('00' + str(self.process_number))[-3:]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_line = 500

    def parse_file(self):
        buffer = []
        logger.warning(self.file_path_worker(torch.utils.data.get_worker_info().id))
        with open(self.file_path_worker(torch.utils.data.get_worker_info().id), encoding='utf-8') as f:
            for i, line in enumerate(f):
                if (len(line) > 0 and not line.isspace()):
                    buffer.append(line.rstrip())
                if (i + 1) % self.buffer_line == 0:
                    buffer = self.tokenizer(buffer, add_special_tokens=True, truncation=False)['input_ids']
                    random.shuffle(buffer)
                    buffer = [item for sublist in buffer for item in sublist]
                    buffer = [buffer[i:i + self.block_size] for i in range(0, len(buffer), self.block_size)]
                    if len(buffer[-1]) != self.block_size:
                        buffer = buffer[:-1]
                    for example in buffer:
                        yield {"input_ids": torch.tensor(example, dtype=torch.long)}
                    buffer = []
            if buffer:
                print('end of file' + self.file_path_worker(torch.utils.data.get_worker_info().id))
                buffer = self.tokenizer(buffer, add_special_tokens=True, truncation=False)['input_ids']
                random.shuffle(buffer)
                buffer = [item for sublist in buffer for item in sublist]
                buffer = [buffer[i:i + self.block_size] for i in range(0, len(buffer), self.block_size)]
                if len(buffer[-1]) != self.block_size:
                    buffer = buffer[:-1]
                for example in buffer:
                    yield {"input_ids": torch.tensor(example, dtype=torch.long)}

    def file_path_worker(self, index):
        return self.file_path + '0' + str(index)

    def __len__(self):
        return 100000000

    def __iter__(self):
        return cycle(self.parse_file())

def main():
    import sys
    logger.warning('max_steps:'+str(int(sys.argv[1])))
    max_steps=int(sys.argv[1])
    eval_steps=int(sys.argv[2])
    logging_steps=int(sys.argv[3])
    save_steps=int(sys.argv[4])
    warmup_steps=int(sys.argv[5])
    per_device_train_batch_size=int(sys.argv[6])
    gradient_accumulation_steps=int(sys.argv[7])
    learning_rate=float(sys.argv[8])
    language=sys.argv[9]
    model_name=sys.argv[10]
    hidden_size=int(sys.argv[11])
    num_attention_heads=int(sys.argv[12])
    num_hidden_layers=int(sys.argv[13])
    max_position_embeddings=int(sys.argv[14])
    wd=sys.argv[15] + '/'
    data_dir=sys.argv[16] + '/'
    training_phase=bool(int(sys.argv[17]))
    pretrained_tokenizer=sys.argv[18]

    number_of_processes = torch.cuda.device_count()

    torch.distributed.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=number_of_processes)

    process_number = torch.distributed.get_rank(group=None)

    logger.warning('process number: ' + str(process_number))

    splits = ['train', 'dev', 'test'] if training_phase else ['test']
    paths = {}
    for v in ['data', 'model', 'tokenizer', 'tensorboard']:
        root = data_dir if v == 'data' or v == 'tokenizer' else wd
        paths[v] = root + v + '/' + language + '/'
        if not os.path.exists(paths[v]) and process_number == 0: os.mkdir(paths[v])

        if v == 'data':
            paths[v] = {split: paths[v] + 'title_abstract.txt_' + split for split in splits}
            # paths[v] = {split: paths[v] + 'title_abstract_sample.txt_' + split for split in splits}

        if v == 'tokenizer' and pretrained_tokenizer == 'pretrained':
            paths[v] += 'tokenizer.model'

        if (v == 'model' or v == 'tensorboard'):
            paths[v] += model_name
            if not os.path.exists(paths[v]) and process_number == 0: os.mkdir(paths[v])

    if pretrained_tokenizer == 'pretrained':
        tokenizer = SPMTokenizer(paths['tokenizer'])
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    dataset = {}
    for split in splits:
        if split == 'train':
            dataset[split] = LineByLineTextStreamer(
                tokenizer=tokenizer,
                file_path=paths['data'][split] + '_' + model_name,
                block_size=max_position_embeddings,
                process_number=process_number,
            )
        else:
            dataset[split] = LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=paths['data'][split],
                block_size=max_position_embeddings
            )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    from transformers import RobertaConfig
    config = RobertaConfig(
        attention_probs_dropout_prob= 0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        gradient_checkpointing=False,
        hidden_act='gelu',
        hidden_dropout_prob= 0.1,
        hidden_size=hidden_size,
        initializer_range= 0.02,
        intermediate_size=4*hidden_size,
        layer_norm_eps=1e-12,
        max_position_embeddings=max_position_embeddings+2,
        model_type='roberta',
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        pad_token_id=1,
        position_embedding_type= 'absolute',
        transformers_version='4.4.2',
        type_vocab_size=tokenizer.pad_token_id,
        use_cache=True,
        vocab_size=tokenizer.vocab_size
    )

    from transformers import RobertaForMaskedLM
    if training_phase:
        model = RobertaForMaskedLM(config=config)
    else:
        logger.warning('Reloading best model...')
        model = RobertaForMaskedLM.from_pretrained(paths['model'], config=config)

    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import IntervalStrategy
    from transformers import SchedulerType
    training_args = TrainingArguments(
        # resume_from_checkpoint=get_last_checkpoint(paths['model'] + '/'),
        ignore_data_skip=True,
        bf16=True,
        # fp16=True,
        do_train=True,
        do_eval=True,
        # logging_first_step=True,
        eval_steps=eval_steps,
        evaluation_strategy=IntervalStrategy.STEPS,
        prediction_loss_only=True,
        save_strategy=IntervalStrategy.STEPS,
        output_dir=paths['model'],
        overwrite_output_dir=True,
        max_steps=max_steps,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_train_batch_size*2,
        logging_steps=logging_steps,
        save_steps=save_steps,
        dataloader_num_workers=0,
        logging_dir=paths['tensorboard'],
        save_total_limit=10000,
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'] if training_phase else None,
        eval_dataset=dataset['dev'] if training_phase else dataset['test']
    )
    from torch.utils.data import DataLoader
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=2,
                pin_memory=self.args.dataloader_pin_memory,
            )

    from types import MethodType
    trainer.get_train_dataloader = MethodType(get_train_dataloader, trainer)

    if training_phase:
        logger.warning('Starting training...')
        logger.warning('ignore_data_skip=' + str(trainer.args.ignore_data_skip))
        trainer.train(resume_from_checkpoint=get_last_checkpoint(paths['model'] + '/'))

        logger.warning('Testing best model...')
        metrics = trainer.evaluate(dataset['test'], metric_key_prefix='test')
        logger.warning(metrics)
        try:
            import math
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("test_set", metrics)
        trainer.save_metrics("test_set", metrics)

        logger.warning('Saving best model...')
        trainer.save_model(paths['model'])
        logger.warning('Training complete')

    else:
        logger.warning('Testing best model...')
        metrics = trainer.evaluate(dataset['test'], metric_key_prefix='test')
        logger.warning(metrics)
        try:
            import math
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("test_set", metrics)
        trainer.save_metrics("test_set", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()