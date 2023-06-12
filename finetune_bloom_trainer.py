"""Code for moss-sft"""

import os
import copy
import json
import torch
import logging
import argparse
from tqdm import tqdm
import torch.distributed as dist

from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed, get_cosine_schedule_with_warmup

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import evaluate
import math

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, block_size=2048,  data_type='train'):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.block_size = block_size

        self.data = []
        # We do not calculate losses for the meta instruction or results returned by plugins
        # The token spans with label -100, [(span_start, span_end), ...]
        self.no_loss_spans = []

        self.load_data()

    def load_data(self):
        do_beta = True
        logger.info("Loading data...")
        data_file = os.path.join(self.data_dir, f'{self.data_type}_data')
        no_loss_spans_file = os.path.join(self.data_dir, f'{self.data_type}_no_loss_spans')
        if os.path.exists(data_file) and os.path.exists(no_loss_spans_file):
            self.data = torch.load(data_file, map_location='cpu')
            self.no_loss_spans = torch.load(no_loss_spans_file, map_location='cpu')
        else:
            with open(os.path.join(self.data_dir, f'{self.data_type}.jsonl'), 'r') as f:
                
                beta_count =10000
                for index, line in enumerate(f):
                    sample = json.loads(line)
                    if do_beta and index==beta_count:
                        break

                    chat = sample['chat']
                    num_turns = int(sample['num_turns'])

                    meta_instruction = sample['meta_instruction']
                    instruction_ids = self.tokenizer.encode(meta_instruction)
                    assert isinstance(instruction_ids, list) and len(instruction_ids) > 0
                    
                    input_ids = copy.deepcopy(instruction_ids)
                    no_loss_spans = [(0, len(instruction_ids))]

                    for i in range(num_turns):
                        cur_turn_ids = []
                        cur_no_loss_spans = []
                        cur_turn = chat[f'turn_{i+1}']
                        for key, value in cur_turn.items():

                            cur_ids = self.tokenizer.encode(value)

                            if key == 'Tool Responses':
                                # The format tokens (<|Results|>:...<eor>\n) should have losses. 
                                cur_no_loss_spans.append((len(input_ids + cur_turn_ids) + 5, len(input_ids + cur_turn_ids + cur_ids) - 2))    

                            assert isinstance(cur_ids, list) and len(cur_ids) > 0

                            cur_turn_ids.extend(cur_ids)

                        if len(input_ids + cur_turn_ids) > self.block_size:
                            break

                        input_ids.extend(cur_turn_ids)
                        no_loss_spans.extend(cur_no_loss_spans)

                    if len(input_ids) == len(instruction_ids):
                        continue

                    assert len(input_ids) > 0 and len(input_ids) <= self.block_size

                    self.data.append(input_ids)
                    self.no_loss_spans.append(no_loss_spans)
            if not do_beta:
                torch.save(self.data, data_file)
                torch.save(self.no_loss_spans, no_loss_spans_file)

        
        logger.info(f"Load data successfully, total {len(self.data)} training samples")
        #self.data = self.data[:100]
        # first inputid2string
        print("first inputid2string:", self.tokenizer.decode(self.data[0]))
        logger.info(f"first inputid2string:{self.tokenizer.decode(self.data[0])}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        no_loss_spans = copy.deepcopy(self.no_loss_spans[index])
        
        data = torch.tensor(data, dtype=torch.long)
        attn_mask = torch.ones_like(data, dtype=torch.bool)
        label = copy.deepcopy(data)

        for no_loss_span in no_loss_spans:
            label[no_loss_span[0] : no_loss_span[1]] = -100

        return data, attn_mask, label
    
    def collate_fn(self, batch):
        batch_input_ids, batch_attn_mask, batch_labels = [], [], []
        for input_ids, attn_mask, label in batch:
            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
            batch_labels.append(label)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        batch_attn_mask = torch.nn.utils.rnn.pad_sequence(batch_attn_mask, batch_first=True, padding_value=0).to(torch.bool)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        return {"input_ids":batch_input_ids, "attention_mask":batch_attn_mask, "labels":batch_labels}
    

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class SelfDefineArguments:
    # 自定义参数类别
    Using_own_wandb:bool = field(default=False, metadata={'help':'using own wandb'} )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    is_llama:bool = field(default=False, metadata={'help':'using llama'} )
    llama_path:Optional[str] = field(default=None, metadata={'help':' llama model path'} )
    data_dir:Optional[str] = field(default=None, metadata={'help':' ldata_dir'} )
    log_dir:Optional[str] = field(default=None, metadata={'help':' log_dir'} )
    eval_times_per_epoch:int = field(default=1,metadata={'help':'eval times per epoch'})
    block_size:int = field(default=1,metadata={'help':'block_size'})
    #seed:int = field(default=1,metadata={'help':'seed'})

def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,SelfDefineArguments))

    model_args, data_args, training_args, own_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir,exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, 'logger.txt'))
    logger.addHandler(file_handler)

    set_seed(training_args.seed)

    # deepspeed needs to know your gradient accumulation steps before hand, so don't forget to pass it
    # Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=1)
    # deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 2
    #accelerator = Accelerator(mixed_precision='fp16') 

    #if accelerator.is_local_main_process:
    #    writer = SummaryWriter(args.log_dir)
    #    writer.add_hparams(vars(args), {})

    #accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    #tokenizer.eos_token_id = 106068 # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_cache=False)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_cache=False)

    
    # add special token
    special_tokens_dict = {'additional_special_tokens': ['<eoc>','<eoh>','<eom>','<eor>','<eot>','[Human]','[MOSS]','<|Inner Thoughts|>','<|Commands|>',
    '<|Results|>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    #model.transformer.gradient_checkpointing = True


    #assert model.transformer.gradient_checkpointing is True

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if False:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SFTDataset(own_args.data_dir, tokenizer)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    val_dataset = SFTDataset(own_args.data_dir, tokenizer, data_type='val')
    #val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    #num_training_steps = (len(train_dataset) * training_args.num_train_epochs) // training_args.gradient_accumulation_steps
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    #model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)


    #global_step = 0
    #metric = SFTMetric(device=torch.cuda.current_device())
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    model.train()

    #best_acc = -float('inf')
    logger.info(f"total step: {len(train_dataset)}")

    # setting trainer step
    steps_per_epoch = (len(train_dataset))
    eval_steps = steps_per_epoch//own_args.eval_times_per_epoch
    
    val_training_args = vars(training_args)
    val_training_args['eval_steps'] = eval_steps

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=train_dataset.collate_fn,
        compute_metrics=compute_metrics if training_args.do_eval  else None,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        #wandbobj = wandbobj
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if False:
        
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name



if __name__ == '__main__':

    
    train()           
