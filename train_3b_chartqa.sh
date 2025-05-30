#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

#reward:
 #   reward_type: batch
 #   reward_function: ./examples/reward_function/refocus.py:compute_score

python3 -m verl.trainer.main \
    config=examples/tooluse_config_llm.yaml \
    data.train_files=datasets/train_full.parquet \
    data.val_files=datasets/val_full.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=3B_CHARTQA \
    trainer.n_gpus_per_node=2 \
    worker.actor.global_batch_size=256 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    data.rollout_batch_size=512 \
    data.val_batch_size=1024 \
    trainer.save_checkpoint_path=./checkpoints/3b_chartqa
    #worker.reward.reward_type=batch \
    #worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
