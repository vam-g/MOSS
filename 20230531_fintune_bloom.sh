num_machines=1
num_processes=$((num_machines * 8))
machine_rank=0

out_put_dir="20230531bloom-3b"
accelerate launch \
	--config_file ./configs/sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard finetune_bloom.py \
	--model_name_or_path /data/application/leyf/llm_zoo/bloom3b_yj/bloom-3B \
	--data_dir ../data/split_data/ \
	--output_dir ../output/$out_put_dir \
	--log_dir ../output/train_logs/$out_put_dir \
	--n_epochs 10 \
	--train_bsz_per_gpu 4 \
	--eval_bsz_per_gpu 4 \
	--learning_rate 0.000015 \
	--max_seq_len 1024 \
    --eval_times_per_epoch 2
