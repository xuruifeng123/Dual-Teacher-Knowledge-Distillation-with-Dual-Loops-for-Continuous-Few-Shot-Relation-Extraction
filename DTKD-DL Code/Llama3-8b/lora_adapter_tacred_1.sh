CUDA_VISIBLE_DEVICES=7 llamafactory-cli export \
    --model_name_or_path /home/database/llm/llama3/llama3-8b \
    --adapter_name_or_path /home/xurf23/xurf_project/LLM_rebuttal_fewrel/save_dora_tacred_5\
    --template llama3 \
    --finetuning_type lora \
    --use_rslora \
    --export_dir /home/xurf23/xurf_project/LLM_rebuttal_fewrel/dora_tacred_5\
    --export_device cpu \
    --export_legacy_format False