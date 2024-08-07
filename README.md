# FedDQC: Data Quality Control in Federated Instruction-tuning of Large Language Models

This is temporary code space for "Data Quality Control in Federated Instruction-tuning of Large Language Models".

## Setup

````
cd OpenFedLLM
conda create -n fedllm python=3.10
conda activate fedllm
pip install -r requirements.txt
````

## Training


````
sh run_sft.sh
````

## Evaluation

````
sh run_eval_med.sh
````