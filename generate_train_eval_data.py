
import json
import argparse
import sklearn
from sklearn.model_selection import KFold
import os 
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
import re 

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    logger.info(f'load dataset :{data_path}')
    data = []
    with open(data_path, 'r') as f:
        for line in tqdm(f.readlines()):
            data+=(json.loads(line))
    return data 

def split_data(data, k, seed, need_fold, shuffle=True):
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    i = 0
    for train, test in kfold.split(data):
        if i == need_fold:
            print(len(train), len(test))

            train_data = np.array(data)[train].reshape(-1).tolist()

            
            dev_data = np.array(data)[test].reshape(-1).tolist()
            return train_data, dev_data
        i += 1

    print('error!!!!')
def save_data(data, save_path):
    logger.info(f"save_data: {save_path}, total count :{len(data)}")
    with open(save_path, 'w') as f:
        for index, idata in tqdm(enumerate(data)):
            f.write(json.dumps(idata, ensure_ascii=False)+'\n')
            if index==0:
                print(idata)

def convert_kernel(plain_text):
    plain_text = plain_text.replace('<eoa>', '<eom>')
    splitby_moss =re.split('(\[MOSS\])', plain_text)
    isp_index = 0
    chat_list = []
    while (isp_index)<len(splitby_moss):
    
        if splitby_moss[isp_index].startswith('[Human]'):#人类
            #    ispan.replace('[Human]','<|Human|>')
            chat_list.append(splitby_moss[isp_index])
        else:
            assert splitby_moss[isp_index].startswith('[MOSS]')
            chat_list.append(splitby_moss[isp_index])
            # split [human]
            isp_index+=1 #下一个
            mix_list = re.split('(\[Human\].*)', splitby_moss[isp_index])
            assert len(mix_list) in [1,3], mix_list
            # moss
            if len(mix_list)==3:
                chat_list[-1]+=mix_list[0]
                assert mix_list[1].startswith('[Human]')
                chat_list.append(mix_list[1])
            else:
                chat_list[-1]+=mix_list[0]
        isp_index+=1
    return chat_list




def single_sample_formatted(data):
    '''
    将数据的格式转化为fintune_moss.py所需要的格式
    '''
    failed_count = 0

    new_data_list = []
    meta_instruction = "我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！"
    for index, idata in tqdm(enumerate(data)):
        new_data_= {
            "conversation_id":'',
            'meta_instruction':'',
            'num_turns':0,
            'chat':{}
        }
        new_data_['conversation_id'] = str(index)
        new_data_['meta_instruction'] = meta_instruction
        new_data_['num_turns']  = idata['num_turns']
        # chat convert
        single_turn = {}
        try:
            chat_list = convert_kernel(idata['plain_text'])
            assert len(chat_list)%2==0, chat_list
            turn_count = 1
            for i in range(1, len(chat_list)+1, 2):

                single_turn[f'turn_{turn_count}'] = {}
                single_turn[f'turn_{turn_count}']['Human'] = chat_list[i-1]
                single_turn[f'turn_{turn_count}']["Inner Thoughts"] = "<|Inner Thoughts|>: None<eot>\n"
                single_turn[f'turn_{turn_count}']["Commands"] =  "<|Commands|>: None<eoc>\n"
                single_turn[f'turn_{turn_count}']["Tool Responses"]= "<|Results|>: None<eor>\n"
                single_turn[f'turn_{turn_count}']["MOSS"] = chat_list[i]
                turn_count+=1
            new_data_['chat'] = single_turn
            new_data_list.append(new_data_)
            
        except Exception as e:
            #logger.error(e)
            #print('failed_count+=1')
            failed_count+=1

    logging.info(f"丢弃数量：{failed_count}")

    return new_data_list

            







def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path_list', default=[], type=json.loads, help="数据的地址列表")
    parser.add_argument('--output_dir', default=[], type=str, help="输出文件的dir")

    args = parser.parse_args()

    total_data = []
    for idata_path in args.data_path_list:
        total_data +=load_data(idata_path)
    logger.info(f'org data len:{len(total_data)}')
    total_data = single_sample_formatted(total_data)

    logger.info(f'after formatted data len:{len(total_data)}')

    
    
    train_data, dev_data = split_data(total_data,5,2023,0,True)

    os.makedirs(args.output_dir, exist_ok=True)

    save_data(train_data, os.path.join(args.output_dir,'train.jsonl'))
    save_data(dev_data, os.path.join(args.output_dir,'val.jsonl'))








if __name__ == '__main__':
    main()