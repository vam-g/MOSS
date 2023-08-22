import argparse
import os
import platform
import warnings
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import deepspeed
from pathlib import Path
import json

import torch
import torch.distributed as dist
from transformers.generation.utils import logger

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# llama
from llama_test_model.modeling_llama import LLaMAForCausalLM
from llama_test_model.configuration_llama import LLaMAConfig
from llama_test_model.tokenization_llama import LLaMATokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer 
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint



def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
web_q = ['when was the last time the dallas cowboys went to the super bowl?',
 'what is the name of the bengals quarterback?',
 'where did giovanni pierluigi da palestrina live?',
 'what is the boston celtics current roster?',
 'where did hernando cortez die?',
 'when was gardasil released to the public?',
 'what types of government does france have?',
 'what is the official language in china?',
 'where did giovanni da verrazano come from?',
 'what country did john cabot sail for?',
 'where was battle of verdun located?',
 'who married jessica simpson?',
 'where is the mtv headquarters?',
 'what season does haley have her second baby?',
 'what money system does greece use?',
 'what things did thomas edison invent?',
 'where to go in phnom penh cambodia?',
 'what city are the swiss alps in?',
 'what kind of money do they use in russia?',
 'who will michael schumacher drive for in 2013?',
 'what team does terrell owens play for this year?',
 'what is considered eastern canada?',
 'what river did henry hudson sail up?',
 'what is the major language spoken in greece?',
 'who plays faramir in lord of the rings?',
 'what is the best currency to take to egypt 2013?',
 'who did hermione granger marry?',
 'what made angela davis famous?',
 'what college is in greeley colorado?',
 'what disease did anne frank get?',
 'who is the member of rajya sabha?',
 'what style of music did louis armstrong play?',
 'where did george harrison live before he died?',
 'who does david beckham play for in 2012?',
 'where is eu headquarters located?',
 'where did kate middleton go to prep school?',
 "what was woodrow wilson's major accomplishments?",
 'where was selena gomez raised?',
 'where is auburn university at?',
 'what language do egyptians use?',
 'what time is the grand prix starting?',
 'where does the shannon river flow?',
 'what currency do mexico use?',
 'who invented the ford motor company?',
 'what type of art does wassily kandinsky do?',
 'who is lleyton hewitt?',
 'who was reese witherspoon married too?',
 'where was san gabriel arcangel located?',
 'where is the euro used?',
 'what money currency does canada use?',
 'where is canadian county oklahoma?',
 'what language do the people in ghana speak?',
 'what is the china money called?',
 'what empire did maria theresa rule?',
 'what state is george washington university located in?',
 'what was the first newspaper called in australia?',
 "what was one of benjamin franklin's inventions?",
 'what does latin america consist of?',
 'where did the welsh language originate from?',
 'what does china border?',
 'what book did john steinbeck wrote about the people in the dust bowl?',
 'where to stay in south rim grand canyon?',
 'what instrument does justin bieber?',
 'what did thomson discover with his cathode ray tube experiment?',
 'where is madeira?',
 'which country does south africa border 2 the north?',
 'what party was winston churchill in politics?',
 'what team did ronaldo play for?',
 'where was alice paul born?',
 'who did matt barnes married?',
 'what year did the mets win their first world series?',
 'who plays dante falconeri?',
 'what did theodore roosevelt do that brought him to national prominence?',
 'who does sam bradford play for?',
 'what language do they speak in indonesia?',
 'what happened to daddy yankee?',
 'what films did liam neeson star in?',
 'what country did francis drake represent?',
 'where is rome italy located on a map?',
 'what is the zip code for moorpark ca?',
 'where to vacation in italy in august?',
 'what happened after the invasion of normandy?',
 'what kind of government does italy have?',
 "what do the colors on mali's flag represent?",
 'where is kate middleton spending christmas?',
 'where was president chester arthur born?',
 'what do they speak in austria?',
 'what kind of political system does iran have?',
 'what did charles babbage make?',
 'where does fabio aurelio play?',
 'what is australian currency?',
 'who did howie long married?',
 'what type of political system does iran have?',
 'what did henry kissinger do?',
 'what channel is anderson cooper talk show on?',
 'what did jean jacques rousseau write?',
 'when does medicare part d start?',
 'what country did francis drake explored for?',
 'where is bob marley from where was he born?',
 'what part of the world is south africa in?']

web_q = web_q#[:1]

def main():
    

    '''
    init & args
    '''

    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()

    world_size = int(os.getenv("WORLD_SIZE", "1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="", 
                         type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--output_dir", default="", 
                        type=str)
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument('--type', required = True, type=str, choices=['fp16', 'fp32'])
    parser.add_argument('--log_txt_name', required = False, type = str, default = 'QA_log.txt')
    parser.add_argument('--gene_out_csv_filename', required = True, type = str) # example: LLAMA_100Q_A_fp16.csv'

    args = parser.parse_args()
    '''
    data type
    '''
    if args.type =='fp16':
        TYPE = torch.float16
    else:
        TYPE = torch.float32

    model_name = args.model_name

    '''
    variables of all
    '''
    checkpoints_json = "checkpoints.json"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    checkpoint_type_dict = {"type": "BLOOM", "version": 1.0}

    #load model
    
    if False:
        tokenizer = LLaMATokenizer.from_pretrained(args.model_name)
        config = LLaMAConfig.from_pretrained(args.model_name)
        #model = LLaMAForCausalLM.from_pretrained(args.model_name, config=config)
        model = LLaMAForCausalLM.from_config(config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    #model = LLaMAForCausalLM.from_pretrained(args.model_name, config=config)
    #model = AutoModelForCausalLM.from_config(config=config)
    #with deepspeed.OnDevice(dtype=TYPE, device='meta'):
    #model = AutoModelForCausalLM.from_config(config=config, torch_dtype = TYPE)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype = TYPE)
    #model.tie_weights()


    # get number of gpus
    #num_gpus = len(args.gpu.split(","))

    '''
    set logging
    '''

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO, filename=os.path.join(args.output_dir, args.log_txt_name), filemode='a')
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    warnings.filterwarnings("ignore")
    
    '''
    set local checkpoint json file
    '''
    def get_checkpoint_files(model_name_or_path):
        cached_repo_dir = model_name_or_path#get_repo_root(model_name_or_path)

        # extensions: .bin | .pt
        # creates a list of paths from all downloaded files in cache dir
        file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
        return file_list

    # set checkpoint file dict to JSON
    def write_checkpoints_json():
        checkpoint_files = get_checkpoint_files(model_name)
        checkpoint_type_dict['checkpoints'] =  checkpoint_files
        if rank == 0:
            data = checkpoint_type_dict
            json.dump(data, open(checkpoints_json, "w"))
    
    # for normal bloom repo we need to write the checkpoints config file
    write_checkpoints_json()
    dist.barrier()

    

    '''
    set deepspeed inference init
    '''
    # DF-inference
    #model = accelerator.prepare(model)
    #injection_policy = {LlamaDecoderLayer:("self_attn.o_proj", "mlp.down_proj") }
    #injection_policy={BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}

    ds_engine = deepspeed.init_inference(
        model,
        mp_size = world_size,
        dtype = TYPE,
        replace_with_kernel_inject=True,
        #base_dir = model_name,
        #checkpoint=checkpoints_json
        #injection_policy = injection_policy
    )
    model = ds_engine.module

    '''
    infer 
    '''
    res = {'question':[],'ans': []}
    if rank==0:
        meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"

    #prompt = meta_instruction
    if rank==0:
        print("欢迎使用 LLAMA 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    for i_q in tqdm(web_q):
        query = i_q #input("<|Human|>: ")
        #prompt = '<|Human|>: ' + query + '<eoh>'
        # logging
        logger.info('*'*20)
        prompt = query  #meta_instruction+ f"[Human]: {query}<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n[MOSS]: "
        inputs = tokenizer(prompt, return_tensors="pt")
        singlesample_token_id = inputs.input_ids[0]
        singlesample_token = tokenizer.convert_ids_to_tokens(singlesample_token_id)
        if rank==0:
            logger.info(f'input text:{prompt}')
            logger.info(f'input text token id:{singlesample_token_id}')
            logger.info(f'input text attention_mask:{inputs.attention_mask}')
            logger.info(f'input text tokenid2token:{singlesample_token}')
        assert len(singlesample_token)==len(singlesample_token_id)
        for it_id, it_tk in zip(singlesample_token_id, singlesample_token):
            if rank==0:
                logger.info(f'token:{it_tk}---> id:{it_id}\n')

        t_response = ''

        model.eval()
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
            if rank==0:
                logger.info(f'generate text token id:{singlesample_token_id}')
                logger.info(f'generate text tokenid2token:{singlesample_token}')
            assert len(singlesample_token)==len(singlesample_token_id)
            for it_id, it_tk in zip(singlesample_token_id, singlesample_token):
                logger.info(f'token:{it_tk}---> id:{it_id}\n')
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            #prompt += response
            #print(response.lstrip('\n')) #去掉多余的生成
            #t_response = response.lstrip('\n')
            t_response = response
        res['question'].append(query)
        res['ans'].append(t_response)
        if rank==0:
            logger.info(f'time:{datetime.now()}')
            logger.info(f'Question:{query}')
            logger.info(f'Answer:{t_response}')
    if rank==0:
        pd.DataFrame(res).to_csv(os.path.join(args.output_dir, args.gene_out_csv_filename))
    
if __name__ == "__main__":
    main()
