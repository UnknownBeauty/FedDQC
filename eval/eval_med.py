import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers


from typing import Dict, Optional, Sequence
import argparse
from tqdm import tqdm
from functools import partial

import pdb
from peft import (
    get_peft_model,
    PeftConfig,
    prepare_model_for_int8_training,
    PeftModel
)
from datasets import load_dataset
import re
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        # "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ###Answer:"
        " ### Instruction:\n{instruction} {input}\n\n### Response: ###Answer:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    'eval_gen': (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        " ### Instruction:\n{instruction} {input}\n\n### Response:"
    )
}

ISTRUCTION_DICT = {
    "medmcqa": "Given your background as a doctor, please provide your insight in addressing the medical questions based on the patient's account. Analyze the question and answer with the best option.",
    "pubmedqa": "Your role as a doctor requires you to answer the medical questions taking into account the patient's description. Analyze the question given its context. Give both long answer and yes/no decision.",
    'medqa': "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    'qa2choice': "Answer this question truthfully",
    'mathqa': "Choose the correct option for the following math question.",
    'aqua_rat': "Choose the correct option for the following math question."
    }
choices = ["A", "B", "C", "D"]
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def load_model(model_name_or_path, checkpoint_path):

    model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto',torch_dtype=torch.float16,)
    if checkpoint_path != '':
        model = PeftModel.from_pretrained(model, checkpoint_path)
        
    model.print_trainable_parameters()
    return model

def load_data(data_path):
    data = load_dataset("json", data_files=data_path, field='instances')
    data = data['train']
    return data

def construct_spedical_tokens_dict(tokenizer) -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def inference_on_one(input_str: Sequence[str], model, tokenizer, max_gen) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        input_ids=model_inputs.input_ids.cuda(),
        max_new_tokens=max_gen,
        top_k=50
    )

    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]

def benchmark_eval(benchmarks, benchmark_num_samples, max_gen, model, tokenizer):
    model.eval()
    special_tokens_dict = construct_spedical_tokens_dict(tokenizer)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_path_dir_full = {'pubmedqa': 'data/eval/test_PMC.json'}

    if benchmarks:
        data_path_dir = {key: data_path_dir_full[key] for key in benchmarks}

    model.cuda()
    acc = {}
    for key in data_path_dir.keys():
        data = load_data(data_path_dir[key])

        if benchmark_num_samples!=0:
            data = data.select(range(benchmark_num_samples))
        output = [None] * len(data) 
        ground_truth = [None] * len(data)
        full_prompt_list = [None] * len(data)
        for i, sample in tqdm(enumerate(data), total=len(data)):
            sample['instruction'] = ISTRUCTION_DICT[key]
            full_prompt = PROMPT_DICT["prompt_input"].format_map(sample) if key!='aqua_rat' else PROMPT_DICT["eval_gen"].format_map(sample)
            full_prompt_list[i] = full_prompt
            ground_truth[i] = sample['output']

        for i, full_prompt in tqdm(enumerate(full_prompt_list), total=len(full_prompt_list)):
            output_str = inference_on_one(full_prompt, model, tokenizer, max_gen=max_gen)
            output[i] = output_str
            
        if key == 'pubmedqa':
            choice_list = []
            for out in output:
                try:
                    response = re.findall(r'(?i)### Response: ([\s\S]*)</s>', out)[0]
                except:
                    response = "No match found"
                try:
                    choice = re.findall(r'###Answer: (yes|no|Yes|No)', response)
                except:
                    choice = "No match found"

                # Check if the choice list is not empty and then convert the first element to lowercase
                if len(choice) == 0:
                    choice = "No match found"
                else:
                    choice = choice[0]
                
                choice_list.append(choice)
        # check the correctness of the output
        print(choice_list)
        correct = 0
        for i in range(len(choice_list)):
            if choice_list[i].lower() == ground_truth[i].lower():
                correct += 1
        acc[key] = correct/len(choice_list)
        
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_token_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument('--benchmark_eval', action='store_true')
    parser.add_argument('--benchmark_num_samples', type=int, default=0)
    parser.add_argument('--benchmarks', nargs="*")
    parser.add_argument('--max_gen', type=int, default=20)
    parser.add_argument('--auth_token_path', type=str, default='hf_token.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    model = load_model(args.model_name_or_path, args.checkpoint_path)
    transformers.set_seed(args.seed)
    
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_token_length,
        padding_side="right",
        use_fast=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
        
    acc = benchmark_eval(args.benchmarks, args.benchmark_num_samples, args.max_gen, model, tokenizer)
    print(acc)
    
    