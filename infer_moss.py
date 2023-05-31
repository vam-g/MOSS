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
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class SFTDataset(Dataset):
    def __init__(self, data_dir, tokenizer,max_len=1024, data_type='train'):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.max_len = max_len
        self.org_input = []

        self.data = []
    
        # We do not calculate losses for the meta instruction or results returned by plugins
        # The token spans with label -100, [(span_start, span_end), ...]
        self.no_loss_spans = []

        self.load_infer_data()

    def load_infer_data(self):
        logger.info("Loading data...")
        #data_file = os.path.join(self.data_dir, f'{self.data_type}_data')
        #no_loss_spans_file = os.path.join(self.data_dir, f'{self.data_type}_no_loss_spans')
        if False:
            pass
        else:
            meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"
            meta_inputid = self.tokenizer.encode(meta_instruction)
            seq_max_len = self.max_len - len(meta_instruction) - 5

            
            #for ikey,value in org_key.items():
            #    meta_inputid+=self.tokenizer.encode(value)


            with open(os.path.join(self.data_dir, f'predict.txt'), 'r') as f:
                #beta_count =100
                for index, line in enumerate(f):

                    org_key = {
                        'Human': "<|Human|>:"+line+"<eoh>",
                        "InnerThoughts": "<|Inner Thoughts|>: None<eot>\n",
                        "Commands": "<|Commands|>: None<eoc>\n",
                        "Tool Responses": "<|Results|>: None<eor>\n",
                        "MOSS":'<|MOSS|>:'
                    }
                    input_ids = []
                    for ikey,value in org_key.items():
                        input_ids+=self.tokenizer.encode(value)
                   
                    
                    #num_turns = int(sample['num_turns'])

                    #meta_instruction = sample['meta_instruction']
                    #instruction_ids = self.tokenizer.encode('<|Human|>: '+line+"<|MOSS|>:")
                    #len(instruction_ids)
                    
                    self.data.append(meta_inputid+input_ids)
                    self.no_loss_spans.append([])
                    self.org_input.append(line)
            
            #torch.save(self.data, data_file)
            #torch.save(self.no_loss_spans, no_loss_spans_file)

        logger.info(f"Load data successfully, total {len(self.data)} training samples")
        #self.data = self.data[:100]
        print("self.org_input:", self.org_input)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        no_loss_spans = copy.deepcopy(self.no_loss_spans[index])
        org_input = copy.deepcopy(self.org_input[index])
        data = torch.tensor(data, dtype=torch.long)
        attn_mask = torch.ones_like(data, dtype=torch.bool)
        label = copy.deepcopy(data)

        for no_loss_span in no_loss_spans:
            label[no_loss_span[0] : no_loss_span[1]] = -100

        return data, attn_mask, label, org_input
    
    def collate_fn(self, batch):
        batch_input_ids, batch_attn_mask, batch_labels, batch_org_text = [], [], [], []
        for input_ids, attn_mask, label, org_text in batch:
            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
            batch_labels.append(label)
            batch_org_text.append(org_text)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        batch_attn_mask = torch.nn.utils.rnn.pad_sequence(batch_attn_mask, batch_first=True, padding_value=0).to(torch.bool)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        return batch_input_ids, batch_attn_mask, batch_labels,batch_org_text
    

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


    #file_handler = logging.FileHandler(os.path.join(args.output_dir, 'logger.txt'))
    #logger.addHandler(file_handler)

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
    tokenizer.eos_token_id = 106068 # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_cache=False)

    model.transformer.gradient_checkpointing = True
    assert model.transformer.gradient_checkpointing is True

   


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    
    #model.tie_weights()
    # load  model
    model.load_state_dict(torch.load(args.output_dir),strict =False)
    #load_checkpoint_and_dispatch(model, args.output_dir, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)
    
    model.eval()
    

    train_dataset = SFTDataset(args.data_dir, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False, drop_last=False, collate_fn=train_dataset.collate_fn)

    #val_dataset = SFTDataset(args.data_dir, tokenizer, data_type='val')
    #val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    #num_training_steps = (len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    with torch.no_grad():
        for batch_cnt, (input_ids, attention_mask, labels, org_input) in (enumerate(train_dataloader)):
            

            inputs = {"input_ids":input_ids, "attention_mask":attention_mask}
            outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=100)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            

            if accelerator.is_main_process:
                print('org_input:',org_input, 'response:', response)
                logger.info(f"org_input:{org_input} response:, {response}")

        #if accelerator.is_main_process:
        #    pbar_train.close()

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
    #os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
