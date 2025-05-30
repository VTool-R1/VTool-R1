import argparse
import datetime
import json
import os
from io import BytesIO
import random
import re
import base64

import backoff
from openai import OpenAI, RateLimitError
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image

from verl.utils.dataset import RLHFDataset, collate_fn
from verl.utils.tokenizer import get_tokenizer, get_processor
from verl.trainer.config import DataConfig, PPOConfig
from verl.tooluse.chart_data import *
from examples.reward_function.refocus import compute_score

# Set openai API
client = OpenAI()

def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        return None
    
@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=6)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def main():
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument("-gpt", "--model_name", help="Model Name", default="gpt-4o")
    parser.add_argument("-o", "--out_gpt_score_root", 
                        help="Path to the output file for gpt scores", default="gpt_score_output.jsonl")
    
    # Parse arguments
    args = parser.parse_args()
    result_root_path = './results_gpt_no_tool_table'
    model_name = args.model_name
    model_short = model_name

    # Set paths
    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_path = f'{result_root_path}/{model_short}_{timestamp}.jsonl'
    stats_path = f'{result_root_path}/{model_short}_{timestamp}_stats.jsonl'
    score_path = f'{result_root_path}/{model_short}_{timestamp}_scores.jsonl'
    output_img_dir = f'{result_root_path}/{model_short}_{timestamp}_edited_imgs/'
    os.mkdir(output_img_dir)


    # Set model
    processor = get_processor("VTOOL/VTOOL-R1-3B-V3-F")
    tokenizer = get_tokenizer("VTOOL/VTOOL-R1-3B-V3-F")

    # Set Data
    config_path = "../examples/tooluse_config.yaml"

    config = OmegaConf.load(config_path).data
    format_prompt_path = "../examples/format_prompt/chartQA.jinja"
    config.val_batch_size = 20

    val_dataset = RLHFDataset(
        data_path='../datasets/table_test.parquet',
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=format_prompt_path,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
    )

    print(f"Length of val dataset: {len(val_dataloader)}")
    print(f"Batch size {val_dataloader.batch_size}")

    with open("prompt.txt", 'r') as file:
        template_prompt = file.read()


    print('Prompt Template:')
    print(template_prompt)
    print('... end template')

    # store stats
    gpt_scores = []

    # store results
    predicts = []
    ground_truths = []
    queries = []

    """
    Eval Loop
    """
    print("Eval...")
    for entry in tqdm(val_dataloader):
        figure_paths = entry['figure_path']
        images = [Image.open('../' + figure_paths[i]) for i in range(len(figure_paths))]
        #print(entry["prompt"])

        message_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(images[i])}",
                            }
                        },
                        {"type": "text", "text": entry["prompt"][i] + template_prompt },
                    ],
                } 
            ] for i in range(len(figure_paths))
        ]

        outputs = []

        for message in message_batch:
            outputs.append(completions_with_backoff(
                model=model_name,
                messages=message,
            ))
        
        with open(output_path, "a") as out_f:
            for idx in range(len(outputs)):
                result_obj = {
                    'query': entry['query'][idx],
                    'ground_truth': entry['ground_truth'][idx],
                    'model_response': outputs[idx].choices[0].message.content,
                }
                predicts.append(result_obj)
                out_f.write(json.dumps(result_obj)+'\n')
        
    print("Using gpt to score model predicts...")
    for result in tqdm(predicts):
        gpt_score, prediction = compute_acc_from_raw_answer(result['query'], result['ground_truth'], result['model_response'])
        with open(score_path, 'a') as f:
            f.write(str(gpt_score) + '\n')
        gpt_scores.append(gpt_score)
        
    print('GPT Scored accuracy: ', len(list(filter(lambda x: x == 1, gpt_scores))) / len(gpt_scores))

    ### batch eval result
    correct_or_not = []
    for idx, result in enumerate(predicts):
        gt = result['ground_truth']
        model_response_text = result['model_response']
        model_response_choice = re.findall(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', model_response_text)
        if len(model_response_choice) == 0:
            # print('re.findall() failed')
            correct_or_not.append(False)
        else:
            model_response_choice = model_response_choice[-1]
            if model_response_choice.lower() == gt.lower():
                # print(f'correct: {gt}')
                correct_or_not.append(True)
            else:
                # print(f'wrong: {model_response_choice} | gt: {gt}')
                correct_or_not.append(False)
    with open(stats_path, 'a') as f:
        for each in correct_or_not:
            f.write('1\n' if each else '0\n')

    print("Mean Overall Exact Match Score:", sum(correct_or_not) / len(correct_or_not))

if __name__ == '__main__':
    main()