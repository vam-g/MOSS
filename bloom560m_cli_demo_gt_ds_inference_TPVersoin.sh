
#export NCCL_IB_DISABLE=1
# /data/application/leyf/llm_zoo/llama65b

deepspeed --num_gpus 8 bloom560m_cli_demo_gt_ds_inference_TPVersoin.py \
--type 'fp16' \
--model_name /data/application/leyf/llm_zoo/bloomz560m \
--output_dir /data/application/leyf/llm_zoo/bloomz560m \
--gene_out_csv_filename 'bloom560m_100QA_TP2.csv'