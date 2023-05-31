num_machines=1
num_processes=$((num_machines * 8))
machine_rank=0

accelerate launch \
	--config_file ./configs/sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard infer_moss.py \
	--model_name_or_path /root/leyf/poj/bloom3b_yj/bloom-3B \
	--data_dir /root/leyf/poj/mmm/data/ \
	--output_dir /root/leyf/poj/mmm/output/bloom3b-sft-2/177114/zero_pp_rank_0_mp_rank_00_model_states.pt \
	--log_dir ../output/train_logs/bloom3b-sft-2 \
	--n_epochs 10 \
	--train_bsz_per_gpu 1 \
	--eval_bsz_per_gpu 1 \
	--learning_rate 0.000015 \
	--max_seq_len 1024 \
    --eval_times_per_epoch 2 > ../output/train_logs/bloom3b-sft-2/logg.txt