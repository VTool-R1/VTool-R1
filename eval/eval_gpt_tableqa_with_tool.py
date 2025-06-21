import os
from io import BytesIO, StringIO
import random
import re
import datetime
import json
from contextlib import redirect_stdout

f = StringIO()
with redirect_stdout(f):
    help(pow)
s = f.getvalue()

from PIL import Image
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf
import base64
from openai import OpenAI, RateLimitError
from jinja2 import Template
import backoff

from verl.utils.dataset import RLHFDataset, collate_fn
from verl.trainer.config import DataConfig, PPOConfig
from verl.utils.tokenizer import get_tokenizer, get_processor
from verl.tooluse.parse import Parser
from verl.tooluse.tools import *
from verl.tooluse.chart_data import *
from examples.reward_function.refocus import compute_score

# avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# set model to eval here !
model_name = "gpt-4o"
model_short = model_name
result_root_path = './results_tableqa_gpt_with_tooluse'

client = OpenAI()

# set paths
if not os.path.exists(result_root_path):
    os.mkdir(result_root_path)
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
output_path = f'{result_root_path}/{model_short}_{timestamp}.jsonl'
stats_path = f'{result_root_path}/{model_short}_{timestamp}_stats.jsonl'
score_path = f'{result_root_path}/{model_short}_{timestamp}_scores.jsonl'
output_img_dir = f'{result_root_path}/{model_short}_{timestamp}_edited_imgs/'
os.mkdir(output_img_dir)


processor = get_processor("VTOOL/VTOOL-R1-3B-V3-F")
tokenizer = get_tokenizer("VTOOL/VTOOL-R1-3B-V3-F")

"""
Load Dataset and DataLoader
"""
config_path = "../examples/tooluse_config.yaml"

config = OmegaConf.load(config_path).data
format_prompt_path = "../examples/format_prompt/chartQA.jinja"
config.val_batch_size = 1

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

print(len(val_dataloader))

tool_parser = Parser()

"""
Variables
"""
# capture code exec
def display(obj):
    global captured_output
    captured_output = obj
    
def get_tool_context():
    context = {
        "display": display,
        "focus_on_columns_with_mask": focus_on_columns_with_mask,
        "focus_on_rows_with_mask": focus_on_rows_with_mask,
        "focus_on_columns_with_draw": focus_on_columns_with_draw,
        "focus_on_rows_with_draw": focus_on_rows_with_draw,
        "focus_on_columns_with_highlight": focus_on_columns_with_highlight,
        "focus_on_rows_with_highlight": focus_on_rows_with_highlight,
        "focus_on_x_values_with_mask": focus_on_x_values_with_mask,
        "focus_on_y_values_with_mask": focus_on_y_values_with_mask,
        "focus_on_x_values_with_draw": focus_on_x_values_with_draw,
        "focus_on_y_values_with_draw": focus_on_y_values_with_draw,
        "focus_on_x_values_with_highlight": focus_on_x_values_with_highlight,
        "focus_on_y_values_with_highlight": focus_on_y_values_with_highlight,
    }
    return context

# stats
num_tool_calls = 0
num_direct = 0
num_success_tool_calls = 0
num_failed_tool_calls = 0

merged_outputs_eval = []
correct_or_not_eval = []
result_objs = []

"""
Eval Loop
"""
for entry in tqdm(val_dataloader):
    # print(entry.keys())
    # print(entry['raw_prompt_ids'])
    # print(entry['multi_modal_data'])
    # print(entry['input_ids'])
    # print(entry['multi_modal_data'])

    
#     input_ids: torch.Tensor = entry['input_ids'] # (bs, prompt_length)
#     attention_mask: torch.Tensor = entry["attention_mask"]
#     position_ids: torch.Tensor = entry["position_ids"]
#     # raw_prompt_ids: torch.Tensor = entry["raw_prompt_ids"]
#     batch_size = input_ids.size(0)
    metadata_batch = []
    for each_metadata in entry['metadata']: # since metadata or json encodings
        metadata_batch.append(json.loads(each_metadata))
    prompts = entry['prompt']
    figure_paths = entry['figure_path']
    original_images = [Image.open('../' + figure_paths[i]) for i in range(len(figure_paths))]
    prompt_str_list = []

    for query in prompts:
        # steal the template from dataloader
        format_prompt = Template(val_dataset.format_prompt.strip())
        prompt_str = format_prompt.render(content=query)
        prompt_str_list.append(prompt_str)
    
    message_batch = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(original_images[i])}",
                        }
                    },
                    {"type": "text", "text": prompt_str_list[i]},
                ],
            } 
        ] for i in range(len(figure_paths))
    ]

    outputs = []

    
    ### FIRST ROLLOUT
    print('=== starting first rollout ===')
    for message in message_batch:
        outputs.append(completions_with_backoff(
            model=model_name,
            messages=message,
        ).choices[0].message.content)
    print('--- first rollout response ---')
    print(outputs[0])
    print('--- END first rollout response ---')
    
    
    parsed_results = [tool_parser.parse(output) for output in outputs]
    
    edited_images = [] # store edited image for second rollout
    code_exec_errors = [] # store code exec error
    code_exec_stdouts = [] # store code exec stdout
    tool_use_indices = [] # store which entry had successful code exec
    second_rollout_datas = [] # store model input data for second rollout
    parsed_codes = []
        
    # this actually takes some time
    for idx, result in enumerate(parsed_results):
        if not result["status"]:

            if result["error_code"] == "NOTOOL":
                num_direct += 1
            else:
                num_tool_calls += 1
                num_failed_tool_calls += 1
                
            edited_images.append(None)
            code_exec_errors.append(None)
            code_exec_stdouts.append(None)
            parsed_codes.append(None)

            continue

        num_tool_calls += 1

        metadata = metadata_batch[idx]

        # ### keep these for 'partial' data
        # y_values = metadata["y_values"]
        # y_bboxes = metadata["y_bboxes"]
        # headers = y_values  # these are your column names
        # bbox_mapping = {label: bbox for label, bbox in zip(y_values, y_bboxes)}
        # ### end keep

        ### keep these for 'full' data
        if metadata["type"] == "v_bar":
            bbox_mapping = metadata["x_values_bbox"]
        elif metadata["type"] == "h_bar":
            bbox_mapping = metadata["y_values_bbox"]
        ### end keep

        ### execute code
        #code_executor = CodeExecutor("executor")
        code = result["content"]
        #exit_code, output, file_paths = code_executor.execute(result["content"])
        figure_path = figure_paths[idx]

        print('--- parsed code ---')
        print(code)
        print('--- END parsed code ---')
        successful = True
        
        captured_output = None

        context = get_tool_context()

        print("Mapping")
        #print(metadata["x_values_bbox"].keys())
        #print(metadata["y_values_bbox"].keys())

        context["image_1"] = Image.open('../'+figure_path)
        if metadata["type"] == "table":
            context["columns_bbox"] = metadata["columns_bbox"]
            context["rows_bbox"] = metadata["row_starters"]
        else:
            context["columns_bbox"] = bbox_mapping
            context["rows_bbox"] = bbox_mapping

        code_exec_error = None
        code_exec_stdout = None
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(code, context)
        except BaseException as e:
            successful = False
            print('~~~ code error ~~~')
            print(f"{e}")
            print('~~~ END code error ~~~')
            code_exec_error = f"{e}"
        code_exec_stdout = f.getvalue()
        
        # log code exec
        code_exec_errors.append(code_exec_error)
        code_exec_stdouts.append(code_exec_stdout)
        parsed_codes.append(code)

        # if successful code exec
        if successful:
            if captured_output is not None:
                successful = isinstance(captured_output, Image.Image)
                print('~~~ captured output ~~~')
                print(captured_output)
                print('~~~ END captured output ~~~')
                '''try:
                    #self.captured_output.save(f"tmp_vis/{figure_ids[idx]}.png")
                    with open(f"tmp_vis/{figure_ids[idx]}.txt", 'w') as file:
                        file.write(output_texts[idx])
                    #the AI may somehow give dicts which is wrong!!!!!
                except Exception as e:
                    successful = False'''
            else:
                successful = False

        if successful:
            num_success_tool_calls += 1

            edited_images.append(captured_output)

            trim_to_action_end = tool_parser.trim_to_action_end(outputs[idx])

            #we need to add image repsonse here:
            trim_to_action_end += "\nOBSERVATION: Execution success. The output is as follows:"
            trim_to_action_end += "\n<the image outputs of the code is added as the second image>"

            prompt_str = prompt_str_list[idx]
            prompt_str = prompt_str.replace("<image>", "")
            content_list = []
            content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(original_images[idx])}",
                        }
                    },)
            content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(captured_output)}",
                        }
                    })
            content_list.append({"type": "text", "text": prompt_str})
            assistant_content = [{"type": "text", "text": trim_to_action_end}]
            messages = [{"role": "user", "content": content_list}, {"role": "assistant", "content": assistant_content}]

            second_rollout_data = {
                'second_rollout_messages': messages,
            }

            second_rollout_datas.append(second_rollout_data)
            tool_use_indices.append(idx)

        else:
            num_failed_tool_calls += 1
            edited_images.append(None)

        print(f"SUCCESSFUL? {successful}")
        print("------")
            
    second_rollout_batch_dict = collate_fn(second_rollout_datas)
    
    print(f'failed tool call count: {num_failed_tool_calls}')
    print('ready for second rollout')
    # print(second_rollout_batch_dict)
            
    ### SECOND ROLLOUT
    second_outputs = []
    if second_rollout_batch_dict == {}:
        print('=== nothing to do in second rollout ===')
    else:
        print('=== starting second rollout ===')

        for message in second_rollout_batch_dict['second_rollout_messages']:
            second_response = completions_with_backoff(
                model=model_name,
                messages=message,
            )
            second_outputs.append(second_response.choices[0].message.content)
        print('--- second rollout response ---')
        print(second_outputs[0])
        print('--- END second rollout response ---')
    
    ### collate results 
    merged_outputs = []
    second_cnt = 0
    for idx, edited_image in enumerate(edited_images):
        if edited_image == None:  # code exec failed or NO TOOL
            merged_outputs.append(outputs[idx])
        else:
            merged_outputs.append(second_outputs[second_cnt])
            second_cnt += 1
    
    ### batch eval result
    correct_or_not = []
    print(entry['ground_truth'])
    for idx, gt in enumerate(entry['ground_truth']):
        model_response_text = merged_outputs[idx]
        model_response_choices = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)', model_response_text)

        # if no matching final answer is found
        if len(model_response_choices) <= 0:
            print(f'wrong: NO ANSWER FOUND | gt: {gt}')
            correct_or_not.append(False)
            continue

        model_response_choice = model_response_choices[-1]
        if model_response_choice.lower() == gt.lower():
            print(f'correct: {gt}')
            correct_or_not.append(True)
        else:
            print(f'wrong: {model_response_choice} | gt: {gt}')
            correct_or_not.append(False)
    
    ### log results
    with open(output_path, "a") as out_f:
        for idx in range(len(outputs)):
            # write edited img to path
            edited_img_path = None
            if edited_images[idx]:
                edited_img_path = output_img_dir + f"/{entry['figure_id'][idx]}_edited.png"
                edited_images[idx].save(edited_img_path)
            
            result_obj = {
                'first_rollout_response': outputs[idx],
                'second_rollout_response': merged_outputs[idx] if edited_images[idx] else None,
                'model_response': merged_outputs[idx],
                'code': parsed_codes[idx],
                'code_error': code_exec_errors[idx],
                'code_stdout': code_exec_stdouts[idx],
                'original_figure_path': entry['figure_path'][idx],
                'edited_figure_path': edited_img_path,
                'ground_truth': entry['ground_truth'][idx],
                'query': entry['query'][idx],
                'prompt': entry['prompt'][idx],
            }
            result_objs.append(result_obj)
            out_f.write(json.dumps(result_obj)+'\n')
            
    merged_outputs_eval += merged_outputs
    correct_or_not_eval += correct_or_not
    
    print('end batch')
    print(f'logged {len(result_objs)} results')
    
"""
score
"""
gpt_scores = []

for result in tqdm(result_objs):
    gpt_score, prediction = compute_acc_from_raw_answer(result['query'], result['ground_truth'], result['model_response'])
    with open(score_path, 'a') as f:
        f.write(str(gpt_score))
    gpt_scores.append(gpt_score)
    
print('GPT scored acc: ', len(list(filter(lambda x: x == 1, gpt_scores))) / len(gpt_scores))

"""
print stats
"""
stats_obj = {
    'gpt_scored_acc': len(list(filter(lambda x: x == 1, gpt_scores))) / len(gpt_scores),
    'exact_match_acc': sum(correct_or_not_eval) / len(correct_or_not_eval),
    'num_tool_calls': num_tool_calls,
    'num_direct': num_direct,
    'num_success_tool_calls': num_success_tool_calls,
    'num_failed_tool_calls': num_failed_tool_calls
}

print(stats_obj)

with open(stats_path, 'w') as f:
    f.write(json.dumps(stats_obj))
    
