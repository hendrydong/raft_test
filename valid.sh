#!/bin/bash
# Please run this script under project directory.

deepspeed_args="--master_port=11211"   
exp_id=raft_infer_get_samples
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}


mkdir -p ${output_dir} ${log_dir}

# /home/xiongwei/LMFlow/data/hh_rlhf/rlhf_train_prompt
# /home/xiongwei/LMFlow/data/clean_hh_rlhf_uncerrtainty_study/rlhf_eval_prompt
export PYTHONPATH=.
deepspeed ${deepspeed_args} \
          examples/raft_align.py \
          --model_name_or_path $1 \
          --mode "raft_get_samples" \
          --iter 0 \
          --raft_infer_set $2\
          --dataset_path /home/hdongaj/Projects/LMFlow/data/helpful/rlhf/rlhf_eval \
          --raft_batch_size 99999 \
          --output_min_length 195 \
          --output_max_length 198 \
          --output_temperature 1.0\
          --trust_remote_code 1\
          --top_reward_percentage 1\
          --inference_batch_size_per_device 32\
          --learning_rate 1e-5 \
          --lr_scheduler_type "constant" \
          --bf16 \
          --deepspeed configs/ds_config_zero2.json \
          --output_reward_path ${project_dir}/tmp/raft_aligner/reward.txt \
          --output_dir ${output_dir} --overwrite_output_dir \
          --run_name ${exp_id} \
          --num_train_epochs 4 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
          --validation_split_percentage 0 \
          --logging_steps 1 \
          --do_train \
          --ddp_timeout 72000 \
          --save_steps 7777 \
          --dataloader_num_workers 1 \
          --preprocessing_num_workers 12 \
          | tee ${log_dir}/raft_align.log \
          2> ${log_dir}/raft_align.err