import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-name", type=str, default=None)
    parser.add_argument("--lora-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_name, args.lora_path)