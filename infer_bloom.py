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

from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

class SFTInferDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_type='train'):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type

        self.data = []
        # We do not calculate losses for the meta instruction or results returned by plugins
        # The token spans with label -100, [(span_start, span_end), ...]
        self.no_loss_spans = []

        self.load_data()

    def load_data(self):
        logger.info("Loading infer data...")
        # predict file来自于 predict.txt文件，该文件是每行一个文本（暂时考虑单轮对话）
        data_file = os.path.join(self.data_dir, 'predict.txt')
        
        meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"

        with open(data_file, 'r') as f:
            #beta_count =100
            for index, line in enumerate(f):
                #sample = json.loads(line)
                #if index==beta_count:
                #    break
                sample = {
                        "conversation_id": "1",
                        "meta_instruction": meta_instruction,
                        "num_turns": 1,
                        "chat": {
                            "turn_1": {
                                "Human": f"[Human]: {line}<eoh>\n",
                                "Inner Thoughts": "<|Inner Thoughts|>: None<eot>\n",
                                "Commands": "<|Commands|>: None<eoc>\n",
                                "Tool Responses": "<|Results|>: None<eor>\n",
                                "MOSS": "[MOSS]: "
                            },
                        },
                        "category": "harmless_zh"
                    }
                chat = sample['chat']
                num_turns = 1 #int(sample['num_turns'])

                
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

                    if len(input_ids + cur_turn_ids) > 2048:
                        break

                    input_ids.extend(cur_turn_ids)
                    no_loss_spans.extend(cur_no_loss_spans)

                if len(input_ids) == len(instruction_ids):
                    continue

                assert len(input_ids) > 0 and len(input_ids) <= 2048

                self.data.append(input_ids)
                #self.no_loss_spans.append(no_loss_spans)
        
       

        logger.info(f"Load data successfully, total {len(self.data)} training samples")
        #self.data = self.data[:100]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        #no_loss_spans = copy.deepcopy(self.no_loss_spans[index])
        
        data = torch.tensor(data, dtype=torch.long)
        attn_mask = torch.ones_like(data, dtype=torch.bool)
        label = copy.deepcopy(data)

        #for no_loss_span in no_loss_spans:
        #    label[no_loss_span[0] : no_loss_span[1]] = -100

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

        return batch_input_ids, batch_attn_mask, batch_labels
  
class SFTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_type='train'):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type

        self.data = []
        # We do not calculate losses for the meta instruction or results returned by plugins
        # The token spans with label -100, [(span_start, span_end), ...]
        self.no_loss_spans = []

        self.load_data()

    def load_data(self):
        logger.info("Loading data...")
        data_file = os.path.join(self.data_dir, f'{self.data_type}_data')
        no_loss_spans_file = os.path.join(self.data_dir, f'{self.data_type}_no_loss_spans')
        if os.path.exists(data_file) and os.path.exists(no_loss_spans_file):
            self.data = torch.load(data_file, map_location='cpu')
            self.no_loss_spans = torch.load(no_loss_spans_file, map_location='cpu')
        else:
            with open(os.path.join(self.data_dir, f'{self.data_type}.jsonl'), 'r') as f:
                #beta_count =100
                for index, line in enumerate(f):
                    sample = json.loads(line)
                    #if index==beta_count:
                    #    break

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

                        if len(input_ids + cur_turn_ids) > 2048:
                            break

                        input_ids.extend(cur_turn_ids)
                        no_loss_spans.extend(cur_no_loss_spans)

                    if len(input_ids) == len(instruction_ids):
                        continue

                    assert len(input_ids) > 0 and len(input_ids) <= 2048

                    self.data.append(input_ids)
                    self.no_loss_spans.append(no_loss_spans)
            
            torch.save(self.data, data_file)
            torch.save(self.no_loss_spans, no_loss_spans_file)

        logger.info(f"Load data successfully, total {len(self.data)} training samples")
        self.data = self.data[:100]

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

        return batch_input_ids, batch_attn_mask, batch_labels
    

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
    

def train(args):


    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'logger.txt'))
    logger.addHandler(file_handler)

    # deepspeed needs to know your gradient accumulation steps before hand, so don't forget to pass it
    # Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=1)
    # deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 2
    accelerator = Accelerator(mixed_precision='fp16') 

    if accelerator.is_main_process:
        writer = SummaryWriter(args.log_dir)
        writer.add_hparams(vars(args), {})

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    #tokenizer.eos_token_id = 106068 # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_cache=False)

    # add special token
    special_tokens_dict = {'additional_special_tokens': ['<eoc>','<eoh>','<eom>','<eor>','<eot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    #model_path ='/mnt/application/leyf/llm_zoo/mmm/output/20230531bloom-3b/29516/zero_pp_rank_0_mp_rank_00_optim_states.pt'
    #model = model.load_state_dict(torch.load(model_path),strict =True)
    if True:
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        unwrapped_model = accelerator.unwrap_model(model)
        model = load_state_dict_from_zero_checkpoint(unwrapped_model, "/mnt/application/leyf/llm_zoo/mmm/output/20230531bloom-3b")


    model.transformer.gradient_checkpointing = True
    assert model.transformer.gradient_checkpointing is True

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    #no_decay = ["bias", "LayerNorm.weight"]
    
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    #train_dataset = SFTDataset(args.data_dir, tokenizer)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    val_dataset = SFTInferDataset(args.data_dir, tokenizer, data_type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=False, collate_fn=val_dataset.collate_fn)

    #num_training_steps = (len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    global_step = 0
    metric = SFTMetric(device=torch.cuda.current_device())

    #model.train()

    #best_acc = -float('inf')
    #logger.info(f"total step: {len(train_dataloader)}")

    #steps_per_epoch = (len(train_dataloader))
    #eval_steps = steps_per_epoch//args.eval_times_per_epoch
    
        
    if True:
        torch.cuda.empty_cache()
        model.eval() 
        for input_ids, attention_mask, labels in val_dataloader:
            with torch.no_grad():
                #output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                #output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=100)
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            if accelerator.is_main_process:
                print('*'*10+'\n')
                logger.info('*'*10+'\n')
                print('input*'*10+'\n')
                print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                print('response----'*10+'\n')
                print(response)
                #print(f'input:{tokenizer.decode(input_ids[0], skip_special_tokens=True)}, response:{response} \n')
                #logger.info(f'input:{tokenizer.decode(input_ids[0], skip_special_tokens=True)}, response:{response} \n')
    #if global_step % args.save_step != 0:
    #    model.save_checkpoint(args.output_dir, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--model_name_or_path', default='./ckpts/moss-16B-base', type=str)
    
    # Data Args
    parser.add_argument('--data_dir', default='./data/sft', type=str)
    parser.add_argument('--output_dir', default='./ckpts/moss-16B-sft', type=str)
    parser.add_argument('--log_dir', default='./train_logs/moss-16B-sft', type=str)
    
    # Training Args
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=9e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=int)
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument('--eval_times_per_epoch', default=2, type=int)
    

    # Other Args
    parser.add_argument('--save_step', default=3000, type=int)
    parser.add_argument('--eval_step', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()


    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
