from io import BytesIO
import random
import re
import json
import datetime

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

model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

model_short = model_name.split("/")[1]

# set paths
if not os.path.exists('./results_no_tool'):
    os.mkdir('./results_no_tool')
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
output_path = f'./results_no_tool/{model_short}_{timestamp}.jsonl'
gpt_score_path = f'./results_no_tool/{model_short}_{timestamp}_gpt_scores.jsonl'
exact_score_path = f'./results_no_tool/{model_short}_{timestamp}_exact_scores.jsonl'


# load model
llm = LLM(model_name, 
          limit_mm_per_prompt={"image": 1}, 
          tensor_parallel_size=4,
          dtype="bfloat16"
         )
processor = get_processor(model_name)
tokenizer = get_tokenizer(model_name)
sampling_params = SamplingParams(temperature=1.0, top_p=0.99, max_tokens=1024) # TODO: what should we use?

"""
Load Dataset and DataLoader
"""
config_path = "../examples/tooluse_config.yaml"

config = OmegaConf.load(config_path).data
format_prompt_path = "../examples/format_prompt/chartQA.jinja"
config.val_batch_size = 20

val_dataset = RLHFDataset(
    data_path='../datasets/test_full.parquet',
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

print(template_prompt)

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
                    {"type": "image", "image": images[i] },
                    {"type": "text", "text": entry["prompt"][i] + template_prompt },
                ],
            } 
        ] for i in range(len(figure_paths))
    ]
    
    prompts = [processor.apply_chat_template(
                   messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
               ) for messages in message_batch
              ]
    
    image_inputs = [process_vision_info(messages)[0] for messages in message_batch]

    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image_input}} for prompt, image_input in zip(prompts, image_inputs)],
        sampling_params=sampling_params,
    )
    
    with open(output_path, "a") as out_f:
        for idx in range(len(outputs)):
            result_obj = {
                'query': entry['query'][idx],
                'ground_truth': entry['ground_truth'][idx],
                'model_response': outputs[idx].outputs[0].text,
            }
            predicts.append(result_obj)
            out_f.write(json.dumps(result_obj)+'\n')
    
print("Using gpt to score model predicts...")
for result in tqdm(predicts):
    gpt_score, prediction = compute_acc_from_raw_answer(result['query'], result['ground_truth'], result['model_response'])
    with open(gpt_score_path, 'a') as f:
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
with open(exact_score_path, 'a') as f:
    for each in correct_or_not:
        f.write('1\n' if each else '0\n')

print("Mean Overall Exact Match Score:", sum(correct_or_not) / len(correct_or_not))