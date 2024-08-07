import sys
import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from fed import *
from config import get_config, save_config, get_model_config, get_training_args
import pdb
from selection.conpro import filter_by_ConPro
import pickle
from datasets import load_dataset
import json
from eval.eval_med import benchmark_eval
from time import time

# ===== Define the arguments =====
script_args, fed_args, peft_config, auth_yaml = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)


sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    token=auth_yaml["auth_token"],
    output_hidden_states=script_args.test_code
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )


if script_args.checkpoint_path != '':
    model = PeftModel.from_pretrained(model, script_args.checkpoint_path, is_trainable=True)
else:
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# print(f">> Init: fixed param: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.weight'][0]}, learned: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0]}")

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False, padding_side="right", token=auth_yaml["auth_token"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function =====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
orig_local_datasets = copy.deepcopy(local_datasets)


if script_args.use_select:
    print('--------------------- data selection ------------------------------')
    if script_args.val_dataset_path != '':
        val_dataset = load_dataset("json", data_files=script_args.val_dataset_path, field='instances')['train']
        val_dataset = process_val_dataset(val_dataset)
    else:
        val_dataset = load_dataset(os.path.join(script_args.local_data_dir, script_args.dataset_name), field='test')['train']
        val_dataset = process_sft_dataset(script_args.dataset_name, val_dataset, script_args.dataset_sample)
        val_dataset = val_dataset.select(range(200))

    val_grad_dict = None
    save_sorted_idx = []
    save_tuple_list = []
    num_total_data = sum([len(local_datasets[i]) for i in range(len(local_datasets))])
    if fed_args.fed_alg.startswith('local'):
        client_list = [int(fed_args.fed_alg.split('local')[-1])+1]
    else:
        client_list = range(fed_args.num_clients)
    print('client_list: ', client_list)

    for client in tqdm(client_list):
        if script_args.select_mode == 'random':
            random_index_list = np.random.permutation(len(local_datasets[client]))
            sorted_tuple = [(random_index_list[i], i) for i in range(len(local_datasets[client]))]
        elif script_args.select_mode == 'IRA':
            sorted_tuple = filter_by_ConPro(local_datasets[client], model, tokenizer, max_length=script_args.seq_length)
        
        num_selected_data = int(len(local_datasets[client]) * script_args.select_rate)
        sorted_mean_rate_list = [sorted_tuple[i][1] for i in range(len(sorted_tuple))]
        save_sorted_idx.append(sorted_mean_rate_list)
        local_datasets[client] = local_datasets[client].select(sorted_mean_rate_list[:num_selected_data])
        save_tuple_list.append(sorted_tuple)

    print('--------------------- Finish data selection ------------------------------')


for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
            
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)        # -1 is an indicator of not training
            continue
        
        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        if script_args.progessive_get_dataset_this_round:
            sub_dataset, subdata_idx = progressive_get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        else:
            sub_dataset, subdata_idx = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)  # get the required sub-dataset for this round

        # pdb.set_trace()
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6) # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)
        # dataset_cartography()

        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )
        
        start_time = time()
        results = trainer.train()
        training_loss[client].append(results.training_loss)
        end_time = time()

        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Aggregate the local models =====
    global_dict, global_auxiliary = global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round, proxy_dict=proxy_dict, opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict))
    set_peft_model_state_dict(model, global_dict)   # update global model

    # ===== Save the model =====
    if (round+1) % fed_args.num_save_rounds == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
