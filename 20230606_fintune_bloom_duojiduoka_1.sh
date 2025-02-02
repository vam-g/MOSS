num_machines=2
num_processes=$((num_machines * 7))
machine_rank=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7'
out_put_dir="20230606bloom7b1-duojiduoka"
accelerate launch \
	--config_file ./configs/sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard finetune_bloom.py \
	--model_name_or_path /mnt/application/leyf/llm_zoo/bloom7b1 \
	--data_dir ../data/split_data/ \
	--output_dir ../output/$out_put_dir \
	--log_dir ../output/train_logs/$out_put_dir \
	--n_epochs 10 \
	--train_bsz_per_gpu 4 \
	--eval_bsz_per_gpu 4 \
	--learning_rate 0.000015 \
	--max_seq_len 1024 \
    --eval_times_per_epoch 2
