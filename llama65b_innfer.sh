num_machines=1
num_processes=$((num_machines * 8))
machine_rank=0
export NCCL_IB_DISABLE=1
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
out_put_dir="20230602bloom-7b1"
accelerate launch \
	--config_file ./accelerate_ds.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard llama65b_cli_demo_gt.py 