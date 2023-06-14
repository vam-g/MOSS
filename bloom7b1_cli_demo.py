import argparse
import os
import platform
import warnings
import logging


from datetime import datetime

import torch
from transformers.generation.utils import logger

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="../LLM/bloom7b1", 
                    choices=[], type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--output_dir", default="../output/20230606bloom7b1-duojiduoka", 
                     type=str)
args = parser.parse_args()
accelerator = Accelerator(mixed_precision='fp16') 
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")


# set logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO, filename=os.path.join(args.output_dir, 'bloom_QA_log.txt'), filemode='a')
logger = logging.getLogger(__name__)

model_path = args.model_name
#if not os.path.exists(args.model_name):
#    model_path = snapshot_download(args.model_name)

#config = MossConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if num_gpus > 1:  
    print("Waiting for all devices to be ready, it may take a few minutes...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, use_cache=False)

    # add special token
    special_tokens_dict = {'additional_special_tokens': ['<eoc>','<eoh>','<eom>','<eor>','<eot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
      #model = load_checkpoint_and_dispatch(
      #   raw_model, model_path, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16
      # )
    unwrapped_model = accelerator.unwrap_model(model)
    model = load_state_dict_from_zero_checkpoint(unwrapped_model, args.output_dir).cuda()
else: # on a single gpu
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, use_cache=False)
    # add special token
    special_tokens_dict = {'additional_special_tokens': ['<eoc>','<eoh>','<eom>','<eor>','<eot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
      
    unwrapped_model = accelerator.unwrap_model(model)
    model = load_state_dict_from_zero_checkpoint(unwrapped_model, args.output_dir).cuda()


def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
def main():

    meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"

    prompt = meta_instruction
    print("欢迎使用 BLOOM-MOSSData-sft 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    while True:
        query = input("<|Human|>: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            clear()
            prompt = meta_instruction
            continue
        #prompt = '<|Human|>: ' + query + '<eoh>'
        prompt = meta_instruction+ f"[Human]: {query}<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n[MOSS]: "
        inputs = tokenizer(prompt, return_tensors="pt")
        t_response = ''
        with torch.no_grad():
            #print(inputs.input_ids)
            #print( tokenizer.decode(inputs.input_ids.numpy().tolist()[0], skip_special_tokens=True))
            outputs = model.generate(
                inputs.input_ids.cuda(), 
                attention_mask=inputs.attention_mask.cuda(), 
                max_length=200, 
                do_sample=True, 
                top_k=40, #top_p=0.8, temperature=0.7, repetition_penalty=1.02,
                num_return_sequences=1, #eos_token_id=106068,
                pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prompt += response
            print(response.lstrip('\n').split('[Human]')[0]) #去掉多余的生成
            t_response = response.lstrip('\n').split('[Human]')[0]
        # logging
        logger.info('*'*20)
        logger.info(f'time:{datetime.now()}')
        logger.info(f'Question:{query}')
        logger.info(f'Answer:{t_response}')
    
if __name__ == "__main__":
    main()
