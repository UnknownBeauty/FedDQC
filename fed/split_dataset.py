import random
import json
import pdb
from datasets import concatenate_datasets
import numpy as np

def split_dataset(fed_args, script_args, dataset, splited_dataset_path=None):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    random.seed(round)
    print("len(dataset): ", len(dataset))
    if len(dataset) < num2sample:
        num2sample = len(dataset)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    return dataset_this_round, random_idx

def progressive_get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    random.seed(round)
    num_data = len(dataset)
    idx_list = list(range(num_data))
    n_sets = script_args.num_stages

    if not script_args.reverse:
        idx_list.reverse() 
    num_each_proportion = int(len(idx_list) / n_sets)
    
    rounds_per_set = fed_args.num_rounds // n_sets
    
    current_set_idx = round // rounds_per_set
    if current_set_idx >= n_sets:
        current_set_idx = n_sets - 1  

    start_idx = current_set_idx * num_each_proportion
    end_idx = start_idx + num_each_proportion
    selected_idx_this_round = idx_list[start_idx:end_idx]
    
    if len(selected_idx_this_round) < num2sample:
        num2sample = len(selected_idx_this_round)
    
    random_idx = random.sample(selected_idx_this_round, num2sample)
    dataset_this_round = dataset.select(random_idx)
    
    return dataset_this_round, random_idx

