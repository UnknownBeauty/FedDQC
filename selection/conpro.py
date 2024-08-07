import torch
import pdb
import numpy as np
import transformers
from typing import Dict
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


PROMPT_DICT = {'full': ("Below is an instruction that describes a task, paired with an input that provides further context. "
                          "Write a response that appropriately completes the request.\n\n"
                          "### Instruction:\n{}\n\n### Response: {}"),
                "input": ("Below is an instruction that describes a task, paired with an input that provides further context. "
                          "Write a response that appropriately completes the request.\n\n"
                          "### Instruction:\n{}\n\n### Response: "),
                }

def generate_and_tokenize_prompt(tokenizer, full_prompt, user_prompt, cutoff_len=1024, add_eos_token=True):
    # full_prompt = PROMPT_DICT['prompt_output'].format(instruction=instruction_template, input=data_point['input'], output=data_point['output'])
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len, add_eos_token)

    # user_prompt = PROMPT_DICT['prompt_input'].format(instruction=instruction_template, input=data_point['input'])
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, cutoff_len, add_eos_token=add_eos_token)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    if len(tokenized_full_prompt['attention_mask']) != len(tokenized_full_prompt['labels']):
        pdb.set_trace()
    return tokenized_full_prompt

def tokenize(tokenizer, prompt, cutoff_len=1024, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

@torch.no_grad()
def filter_by_ConPro(dataset, model, tokenizer, max_length, reverse):
    condition_loss = []
    uncondition_loss = []
    model.eval()
    for idx, example in enumerate(dataset):
        try:
            full_prompt = PROMPT_DICT['full'].format(example['instruction'], example['response'])
            input_prompt = PROMPT_DICT['input'].format(example['instruction'])
        except:
            full_prompt = PROMPT_DICT['full'].format(example['prompt'], example['rejected'])
            input_prompt = PROMPT_DICT['input'].format(example['prompt'])
        tokenized_conidtion = generate_and_tokenize_prompt(tokenizer, full_prompt, input_prompt, cutoff_len=max_length)
        try:
            tokenized_uncondition = tokenize(tokenizer, example['response'], cutoff_len=max_length, add_eos_token=True)
        except:
            tokenized_uncondition = tokenize(tokenizer, example['rejected'], cutoff_len=max_length, add_eos_token=True)
        pdb.set_trace()
        input_ids_condition = torch.tensor(tokenized_conidtion['input_ids']).unsqueeze(0).to('cuda')
        attention_mask_condition = torch.tensor(tokenized_conidtion['attention_mask']).unsqueeze(0).to('cuda')
        labels_condition = torch.tensor(tokenized_conidtion['labels']).unsqueeze(0).to('cuda')

        input_ids_uncondition = torch.tensor(tokenized_uncondition['input_ids']).unsqueeze(0).to('cuda')
        attention_mask_uncondition = torch.tensor(tokenized_uncondition['attention_mask']).unsqueeze(0).to('cuda')
        labels_uncondition = torch.tensor(tokenized_uncondition['labels']).unsqueeze(0).to('cuda')
        with torch.no_grad():
            try:
                outputs_condition = model(input_ids=input_ids_condition, attention_mask=attention_mask_condition, labels=labels_condition)
                outputs_uncondition = model(input_ids=input_ids_uncondition, attention_mask=attention_mask_uncondition, labels=labels_uncondition)
            except:
                pdb.set_trace()
        condition_loss.append(outputs_condition.loss.item())
        uncondition_loss.append(outputs_uncondition.loss.item())
    

    # conpro=uncondition-condition
    conpro = [(uncondition_loss[i]-condition_loss[i], i) for i in range(len(condition_loss))]
    # conpro = [(condition_loss[i]-uncondition_loss[i], i) for i in range(len(condition_loss))]
    sorted_tuple = sorted(conpro, reverse=reverse)

    return sorted_tuple
