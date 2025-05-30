#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

#reward:
 #   reward_type: batch
 #   reward_function: ./examples/reward_function/refocus.py:compute_score

python3 -m verl.trainer.main \
    config=examples/tooluse_config_llm.yaml \
    data.train_files=datasets/table_train.parquet \
    data.val_files=datasets/table_test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=7B_TABLE \
    worker.actor.global_batch_size=32 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.nnodes=2 \
    trainer.save_checkpoint_path=./checkpoints/7b_table
    #worker.reward.reward_type=batch \
    #worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
