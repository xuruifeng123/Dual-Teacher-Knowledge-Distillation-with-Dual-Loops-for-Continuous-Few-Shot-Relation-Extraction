export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/xurf23/xurf_project/LLM_rebuttal_fewrel/dora_tacred_5 \
    --dataset tacred_train_5_8 \
    --dataset_dir /home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data \
    --template llama3 \
    --finetuning_type lora \
    --use_rslora \
    --output_dir /home/xurf23/xurf_project/LLM_rebuttal_fewrel/save_dora_tacred_5\
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --plot_loss \
    --fp16