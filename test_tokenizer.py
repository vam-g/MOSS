from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer= AutoTokenizer.from_pretrained("/root/leyf/poj/mmm/PLM", trust_remote_code=True)
tokenizer.eos_token_id = 106068
org_text = '你好么！！！！'

input_id = tokenizer.encode(org_text)
input_id= [52746,  54375,  51231,  47797,    121,  59849,  25082,      0,   5525,
          50263,  51053,  90245,  51724,  50479,  50884,  50840,  50257,  51069,
          69362,  52611,  52794,  53647,  50257,  38834,  53089,  51201,  55627,
          51743,  50260,  66218,  52530,  74790,    163,  18433,  50257,  81802,
          59602,  50422,     58,  20490,   5974,  10263,     99,    224,  50581,
          57772,  76883,  58740,  61978,  50379, 106067,    220,     27,     91,
            818,   1008,  33058,     91,  31175,   6045, 106069,    198,     27,
             91,   6935,   1746,     91,  31175,   6045, 106070,    198]
output_res = tokenizer.decode(input_id)
print(org_text,input_id,  output_res)