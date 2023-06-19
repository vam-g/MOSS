import argparse
import os
import platform
import warnings
import logging


from datetime import datetime

import torch
from transformers.generation.utils import logger

#from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

# llama
from llama_test_model.modeling_llama import LLaMAForCausalLM
from llama_test_model.configuration_llama import LLaMAConfig
from llama_test_model.tokenization_llama import LLaMATokenizer

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="/mnt/application/leyf/llm_zoo/llama7b", 
                    choices=[], type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--output_dir", default="/mnt/application/leyf/llm_zoo/llama7b", 
                     type=str)
args = parser.parse_args()
#accelerator = Accelerator(mixed_precision='fp16') 
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

# set logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO, filename=os.path.join(args.output_dir, 'QA_log.txt'), filemode='a')
logger = logging.getLogger(__name__)

tokenizer = LLaMATokenizer.from_pretrained(args.model_name)
config = LLaMAConfig.from_pretrained(args.model_name)
model = LLaMAForCausalLM.from_pretrained(args.model_name, config=config)

#model.load_state_dict(torch.load(os.path.join(args.model_name,'pytorch_model.bin'), map_location=torch.device('cpu')),strict =False)

model_path = args.model_name
model.cuda()




def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
def main():

    meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"

    prompt = meta_instruction
    print("欢迎使用 LLAMA 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    while True:
        query = input("<|Human|>: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            clear()
            prompt = meta_instruction
            continue
        #prompt = '<|Human|>: ' + query + '<eoh>'
        # logging
        logger.info('*'*20)
        prompt = query  #meta_instruction+ f"[Human]: {query}<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n[MOSS]: "
        inputs = tokenizer(prompt, return_tensors="pt")
        singlesample_token_id = inputs.input_ids[0]
        singlesample_token = tokenizer.convert_ids_to_tokens(singlesample_token_id)
        logger.info(f'input text:{prompt}')
        logger.info(f'input text token id:{singlesample_token_id}')
        logger.info(f'input text attention_mask:{inputs.attention_mask}')
        logger.info(f'input text tokenid2token:{singlesample_token}')
        assert len(singlesample_token)==len(singlesample_token_id)
        for it_id, it_tk in zip(singlesample_token_id, singlesample_token):
            logger.info(f'token:{it_tk}---> id:{it_id}\n')

        t_response = ''
        with torch.no_grad():
            #print(inputs.input_ids)
            #print( tokenizer.decode(inputs.input_ids.numpy().tolist()[0], skip_special_tokens=True))
            outputs = model.generate(
                inputs.input_ids.cuda(), 
                attention_mask=inputs.attention_mask.cuda(), 
                max_length=200, 
                do_sample=False, 
                top_k=1, #top_p=0.8, temperature=0.7, repetition_penalty=1.02,
                num_return_sequences=1, #eos_token_id=106068,
                pad_token_id=tokenizer.pad_token_id)
            
            # generate
            
            singlesample_token_id = outputs[0]
            singlesample_token = tokenizer.convert_ids_to_tokens(singlesample_token_id)
            
            logger.info(f'generate text token id:{singlesample_token_id}')
            logger.info(f'generate text tokenid2token:{singlesample_token}')
            assert len(singlesample_token)==len(singlesample_token_id)
            for it_id, it_tk in zip(singlesample_token_id, singlesample_token):
                logger.info(f'token:{it_tk}---> id:{it_id}\n')
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prompt += response
            print(response.lstrip('\n')) #去掉多余的生成
            t_response = response.lstrip('\n')
        
        logger.info(f'time:{datetime.now()}')
        logger.info(f'Question:{query}')
        logger.info(f'Answer:{t_response}')
    
if __name__ == "__main__":
    main()
