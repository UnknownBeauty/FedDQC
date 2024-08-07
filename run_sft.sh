
##################################### FedDQC #####################################
dataset_name=bad_pub_8k_ex50
output_dir=./output
num_clients=5
sample_clients=2


use_select=False
select_rate=0.5
select_mode='IRA'
filter_low=True
num_stages=3
reverse=True
progessive_get_dataset_this_round=False

local_data_dir=data

fed_alg="fedavg"
max_steps=10
batch_size=4
gradient_accumulation_steps=4
seq_length=1024
num_rounds=100
num_save_rounds=100

lr=1e-4
peft_lora_r=64
peft_lora_alpha=128

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=1 python main_sft.py \
 --learning_rate $lr \
 --local_data_dir $local_data_dir \
 --dataset_name $dataset_name \
 --fed_alg $fed_alg \
 --max_steps $max_steps \
 --save_steps $max_steps \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft --load_in_4bit \
 --output_dir $output_dir \
 --peft_lora_r $peft_lora_r \
 --peft_lora_alpha $peft_lora_alpha \
 --use_select $use_select \
 --select_rate $select_rate \
 --select_mode $select_mode \
 --filter_low $filter_low \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --num_rounds $num_rounds \
 --num_save_rounds $num_save_rounds \
 --reverse $reverse \
 --progessive_get_dataset_this_round $progessive_get_dataset_this_round \
 --num_stages $num_stages \
 --trust_remote_code True \


##################################### FedDQC #####################################
dataset_name=bad_pub_8k_ex50
output_dir=./output
num_clients=5
sample_clients=2


use_select=True
select_rate=0.5
select_mode='IRA'
filter_low=True
num_stages=3
reverse=True
progessive_get_dataset_this_round=True

local_data_dir=data

fed_alg="fedavg"
max_steps=10
batch_size=4
gradient_accumulation_steps=4
seq_length=1024
num_rounds=100
num_save_rounds=100

lr=1e-4
peft_lora_r=64
peft_lora_alpha=128

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=1 python main_sft.py \
 --learning_rate $lr \
 --local_data_dir $local_data_dir \
 --dataset_name $dataset_name \
 --fed_alg $fed_alg \
 --max_steps $max_steps \
 --save_steps $max_steps \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft --load_in_4bit \
 --output_dir $output_dir \
 --peft_lora_r $peft_lora_r \
 --peft_lora_alpha $peft_lora_alpha \
 --use_select $use_select \
 --select_rate $select_rate \
 --select_mode $select_mode \
 --filter_low $filter_low \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --num_rounds $num_rounds \
 --num_save_rounds $num_save_rounds \
 --reverse $reverse \
 --progessive_get_dataset_this_round $progessive_get_dataset_this_round \
 --num_stages $num_stages \
 --trust_remote_code True \
