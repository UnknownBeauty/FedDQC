CUDA_VISIBLE_DEVICES=0 python eval/eval_med.py \
    --checkpoint_path CHECKPOINT_PATH \
    --max_token_length 1024 \
    --max_gen 20 \
    --benchmarks 'pubmedqa' \
