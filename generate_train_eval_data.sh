

data_path_list='["/root/leyf/poj/mmm/data/zh_helpfulness.json","/root/leyf/poj/mmm/data/zh_honesty.json"]'
python generate_train_eval_data.py --data_path_list $data_path_list --output_dir '../data/split_data/'