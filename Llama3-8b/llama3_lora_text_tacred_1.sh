export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/xurf23/xurf_project/llama3/llama3-8b \
    --adapter_name_or_path /home/xurf23/xurf_project/LLM_rebuttal_fewrel/save_dora_tacred_5  \
    --eval_dataset tacred_test_5_1 \
    --dataset_dir /home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir /home/xurf23/xurf_project/LLM_rebuttal_fewrel/tacred_dora_out \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --flash_attn auto\
    --max_new_tokens 128 \
    --fp16 \
    