num_machines=1
num_processes=$((num_machines * 1))
machine_rank=0

accelerate launch \
	--config_file ./configs/sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard infer_bloom.py \
	--model_name_or_path /mnt/application/leyf/llm_zoo/bloom7b1 \
	--data_dir ../data/ \
	--output_dir ../output/20230606bloom7b1-duojiduoka \
	--log_dir ../output/train_logs/20230606bloom7b1-duojiduoka \
	--n_epochs 10 \
	--train_bsz_per_gpu 1 \
	--eval_bsz_per_gpu 1 \
	--learning_rate 0.000015 \
	--max_seq_len 1024 \
    --eval_times_per_epoch 2
    # > ../output/train_logs_files/20230531bloom-3b_sft_weight.txt