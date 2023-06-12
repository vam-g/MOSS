export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7'
output_dir=20230611_trainer_bloom
export WANDB_PROJECT=$output_dir
python -m torch.distributed.run --nproc_per_node=7 --nnode=2 --node_rank=0 \
    --master_addr "wxhd10" \
    --master_port=9901 \
    finetune_bloom_trainer.py \
	--model_name_or_path /mnt/application/leyf/llm_zoo/bloom3b_yj/bloom-3B \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --num_train_epochs 10 \
	--data_dir ../data/split_data/ \
	--output_dir ../output/$output_dir \
	--log_dir ../output/train_logs/$output_dir \
    --logging_strategy=steps \
    --logging_steps=1 \
    --save_total_limit=2 \
    --log_on_each_node=False \
    --logging_nan_inf_filter=False \
    --gradient_accumulation_steps=1 \
	--per_device_train_batch_size 1 \
	--learning_rate 0.000015 \
	--block_size 2048 \
    --eval_times_per_epoch 2 \
    --seed 2023 \
    --report_to wandb \
    --gradient_checkpointing=True \
    --deepspeed ./configs/trainer_ds.json 