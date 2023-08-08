export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
output_dir=20230809_ds_trainer_baichuan7b_
export WANDB_PROJECT=$output_dir
python -m torch.distributed.run --nproc_per_node=8 --nnode=1 --node_rank=0 \
    --master_addr "wxhd10" \
    --master_port=9902 \
    finetune_baichuan7b_trainer.py \
	--model_name_or_path /mnt/application/leyf/llm_zoo/baichuan7B \
    --do_train \
    --overwrite_output_dir \
    --num_train_epochs 3 \
	--data_dir ../bloom_data/split_data/ \
	--output_dir ../output/$output_dir \
	--log_dir ../output/train_logs/$output_dir \
    --evaluation_strategy='no' \
    --logging_strategy=steps \
    --logging_steps=1 \
    --save_total_limit=2 \
    --log_on_each_node=False \
    --logging_nan_inf_filter=False \
    --gradient_accumulation_steps=4 \
	--per_device_train_batch_size 4 \
	--learning_rate 0.000015 \
	--block_size 512 \
    --fp16 \
    --fp16_full_eval \
    --seed 2023 \
    --gradient_checkpointing=True \
    --deepspeed ./configs/trainer_ds_baichuan7b.json &> ../output/train_logs/$output_dir/training.log