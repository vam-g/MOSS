import argparse
import os
import platform
import warnings
import logging
import json
import re

from datetime import datetime

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers.generation.utils import logger

#from models.configuration_moss import MossConfig
#from models.modeling_moss import MossForCausalLM
#from models.tokenization_moss import MossTokenizer
from utils import StopWordsCriteria
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint




model_path = '/mnt/application/leyf/llm_zoo/mmm/output/20230606bloom7b1-duojiduoka'
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="/data/application/leyf/llm_zoo/bloom7b1", 
                    choices=["fnlp/moss-moon-003-sft", 
                             "fnlp/moss-moon-003-sft-int8", 
                             "fnlp/moss-moon-003-sft-int4"], type=str)
parser.add_argument("--gpu", default="0", type=str)

# /mnt/application/leyf/llm_zoo/mmm/output/20230606bloom7b1-duojiduoka
# /mnt/application/leyf/ds_chat/rlhf_output/20230726__newpara_1ppepoch_add_specialtoken_bloom7b1_from_sft/actor/
# /mnt/application/leyf/llm_zoo/bloomz560m
parser.add_argument("--output_dir", default=model_path, 
                     type=str)
args = parser.parse_args()
#accelerator = Accelerator(mixed_precision='fp16') 
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

if args.model_name in ["fnlp/moss-moon-003-sft-int8", "fnlp/moss-moon-003-sft-int4"] and num_gpus > 1:
    raise ValueError("Quantized models do not support model parallel. Please run on a single GPU (e.g., --gpu 0) or use `fnlp/moss-moon-003-sft`")

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")


# set logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO, filename=os.path.join(args.output_dir, '20230821bloom_QA_log.txt'), filemode='w')
logger = logging.getLogger(__name__)

model_path = args.model_name
#if not os.path.exists(args.model_name):
#    model_path = snapshot_download(args.model_name)

#config = MossConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
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
    #unwrapped_model = accelerator.unwrap_model(model)
    model.load_state_dict(torch.load(os.path.join(args.output_dir,'pytorch_model.bin'), map_location=torch.device('cuda')),strict =True)
else: # on a single gpu
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, use_cache=False)
    # add special token
    special_tokens_dict = {'additional_special_tokens': ['<eoc>','<eoh>','<eom>','<eor>','<eot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(250685) # 250685
      
    #unwrapped_model = accelerator.unwrap_model(model)
    model.load_state_dict(torch.load(os.path.join(args.output_dir,'pytorch_model.bin'), map_location=torch.device('cuda')),strict =True)

#model.resize_token_embeddings(250880)
#print('model,', model)

model.cuda()
#torch.save(model.state_dict(), os.path.join( args.output_dir, "pytorch_model.bin"))

def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

name = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography', 'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']


from datasets import load_dataset

def get_dataset(iname):
    
    dataset=load_dataset(r"ceval/ceval-exam",name=iname)
    #print(dataset)
    return dataset
    

def get_res_from_str(res_str):
    # 通过re从答案中抽取A/B/C/D选项
    pattern = r'[A-Za-z]'
    match = re.search(pattern, res_str)
    if match:
        res_str = match.group()
        
        if res_str  not in ['A','B','C','D']:
            return 'A'
        return res_str
    else:
        return 'A'



def main():

    meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"
    out_put_path ='../bloom7b_ceval_sft.json'
    #保存结果
    res_map = {}
    prompt = meta_instruction
    print("欢迎使用 MOSS 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    #if True:
    for iname in name:
        dataset = get_dataset(iname)
        res_map[iname]={}

        test_ds = dataset['test']
        #clo_name = test_ds['test'].columns_name
        #clo_name = 
        def get_prompt_func(example):
            #print(dataset['val'][0])
            # {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}
            #print('example:', example)
            #print("example['A']:", example['A'])
            question=f'''
            以下是中国关于{iname}考试的单项选择题，请选出其中的正确答案。
            '''+example['question']+'''\nA. '''+example['A'] \
            +'''\nB. '''+example['B'] \
            +'''\nC. '''+example['C'] \
            +'''\nD. '''+example['D']
            
            prompt = meta_instruction+ f"[Human]: "+question+"<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n[MOSS]: "

            logger.info('*'*20)
            logger.info(f'question:{prompt}')
            logger.info('*'*20)
            print('prompt:', prompt)
            #res = tokenizer(prompt,truncation='only_first' ,max_length=200, padding='max_length', return_tensors="pt")
            #print("res.input_ids.shape:", res.input_ids.shape)
            #print("res.attention_mask.shape:", res.attention_mask.shape)
            result ={}
            result['prompt'] = prompt
            return result #res.input_ids, res.attention_mask

        #def tokenizing_func(example):
        def coll_fn(examples):
            # TODO coll_fn操作
            #print('coll_fn:', examples)
            prompt_list = []
            index_list = []
            for iex in examples:
                prompt_list.append(iex['prompt'])
                index_list.append(str(iex['id']))
            res = tokenizer(prompt_list,truncation='only_first' ,max_length=200, padding='max_length', return_tensors="pt")
            return res.input_ids.cuda(), res.attention_mask.cuda(),prompt_list, index_list


        tokenizerd_ds = test_ds.map(get_prompt_func, batched=False, num_proc=4)
        dataloader = DataLoader(tokenizerd_ds, batch_size=4, shuffle= False, collate_fn=coll_fn)

        for input_ids, attention_mask, prompt_list,index_list in (dataloader):
            #print(f'torch.cuda.is_available():{torch.cuda.is_available()}')
            batch_todevice ={}
            #print("input_ids, attention_mask:", input_ids, attention_mask)

            batch_todevice['input_ids'] = input_ids
            batch_todevice['attention_mask'] = attention_mask

            with torch.no_grad():
                #print(inputs.input_ids)
                #print( tokenizer.decode(inputs.input_ids.numpy().tolist()[0], skip_special_tokens=True))
                outputs = model.generate(
                    **batch_todevice,
                    max_new_tokens=20, 
                    do_sample=False, 
                    top_k=1, 
                    temperature=1,
                    num_return_sequences=1, #eos_token_id=106068,
                    pad_token_id=tokenizer.pad_token_id)
                
                print("outputs shape", outputs.shape)
                 
                print('output_id:',outputs.shape, outputs[:,batch_todevice['input_ids'].shape[1]:])
                output_valid = outputs[:,batch_todevice['input_ids'].shape[1]:]
                
                response =  []
                for iout in output_valid:
                    ires = tokenizer.decode(iout, skip_special_tokens=True)
                    ires = ires.lstrip('\n').split('[Human]')[0]
                    response.append(get_res_from_str(ires))
                #prompt += response
                #print(response.lstrip('\n').split('[Human]')[0]) #去掉多余的生成
                #t_response = response.lstrip('\n').split('[Human]')[0]
            #print('question:', prompt_list)
            #print('response:', response)
            for iq, ires in zip(prompt_list, response):
                print('****\n')
                print('q:', iq)
                print('ires:', ires)
            # add_Res
            for index_str, ires in zip(index_list, response):
                res_map[iname][index_str] = ires


        # logging
        logger.info('*'*20)
        logger.info(f'time:{datetime.now()}')
        #logger.info(f'Question:{query}')
        #logger.info(f'Answer:{t_response}')
    # save:
    with open(out_put_path, 'w') as f:
        json.dump(res_map, f)

    
if __name__ == "__main__":
    main()
