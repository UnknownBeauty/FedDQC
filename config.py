from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime  
import pytz
import yaml


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    num_save_rounds: Optional[int] = field(default=50, metadata={"help": "the number of save rounds"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    heterogeneity_param: Optional[float] = field(default=3, metadata={"help": "the heterogeneity parameter"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fed_momentum: Optional[float] = field(default=0.5, metadata={"help": "the momentum parameter at the server"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-2, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    auth_token_path: Optional[str] = field(default="./hf_token.yaml", metadata={"help": "the path to the auth token"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=30000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory"})
    use_select: Optional[bool] = field(default=False, metadata={"help": "Wether to use data selection or not"})
    select_rate: Optional[float] = field(default=0.7, metadata={"help": "the rate of data selection"})
    select_mode: Optional[str] = field(default="random", metadata={"help": "the mode of data selection"})
    select_idx_path: Optional[str] = field(default='', metadata={"help": "the path of selected data index"})
    val_dataset_path: Optional[str] = field(default='', metadata={"help": "the path of validation dataset"})
    checkpoint_path: Optional[str] = field(default='', metadata={"help": "the path of checkpoint"})
    bad_idx_path: Optional[str] = field(default='', metadata={"help": "the path of bad index"})
    progessive_get_dataset_this_round: Optional[bool] = field(default=False)
    reverse: Optional[bool] = field(default=False)
    filter_low: Optional[bool] = field(default=False)
    num_stages: Optional[int] = field(default=3)
    


parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_attn", "c_proj", "w1", "w2"]
    )
else:
    peft_config = None

def get_config():
    with open(script_args.auth_token_path, "r") as f:
        auth_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return script_args, fed_args, peft_config, auth_yaml

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args):
    beijing_tz = pytz.timezone('Asia/Beijing')
    beijing_time = (datetime.now(beijing_tz)).strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(script_args.dataset_name)
    if script_args.progessive_get_dataset_this_round:
        script_args.output_dir = f"{script_args.output_dir}/progress_{script_args.select_mode}_{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{beijing_time}"
    elif script_args.use_select:
        script_args.output_dir = f"{script_args.output_dir}/select_{script_args.select_mode}_{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{beijing_time}"
    else:
        script_args.output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{beijing_time}"
    os.makedirs(script_args.output_dir, exist_ok=True)
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)