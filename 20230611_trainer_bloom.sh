python finetune_bloom_trainer.py \
	--model_name_or_path /mnt/application/leyf/llm_zoo/bloom3b_yj/bloom-3B \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --num_train_epochs 10 \
	--data_dir ../data/split_data/ \
	--output_dir ../output/20230611_trainer_bloom \
	--log_dir ../output/train_logs/20230611_trainer_bloom \
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
    --seed 2023 