num_machines=1
num_processes=$((num_machines * 8))
machine_rank=0
export NCCL_IB_DISABLE=1
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

accelerate launch \
	--config_file ./configs/accelerate_ds.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard llamacode34b_cli_demo_ds_zero3_DDP.py \
	--model_name /mnt/application/leyf/llm_zoo/llama_code_7b_python \
	--output_dir /mnt/application/leyf/llm_zoo/llama_code_7b_python