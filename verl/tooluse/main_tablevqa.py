import json
import os
import argparse
from pathlib import Path
from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt_need import TablePrompt_mix_col_row_cot, python_codes_for_table_bbox_load, MULTIMODAL_ASSISTANT_MESSAGE
from parse import Parser
from execution import CodeExecutor
from utils import custom_encoder
from chart_data import compute_acc_single
from tqdm import tqdm
import shutil

os.environ["AUTOGEN_USE_DOCKER"] = "False"

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError

def run_agent(task_directory, ex, prompt_generator, MAX_REPLY=1, system_prompt=MULTIMODAL_ASSISTANT_MESSAGE, overwrite=False):
    if not overwrite and os.path.exists(os.path.join(task_directory, "answer.json")):
        outputs = json.load(open(os.path.join(task_directory, "answer.json"), 'r'))
        return outputs['acc']

    parser = Parser()
    executor = CodeExecutor(working_dir=task_directory)

    column_headers = ex["column_headers"]
    raw_columns_bbox = ex["columns_bbox"]
    if column_headers:
        columns_bbox = {column_headers[i]: raw_columns_bbox[i] for i in range(len(column_headers))}
    else:
        columns_bbox = {}
    row_starters = ex["row_starters"]
    raw_rows_bbox = ex["rows_bbox"]
    if row_starters:
        rows_bbox = {row_starters[i]: raw_rows_bbox[i] for i in range(len(row_starters))}
    else:
        rows_bbox = {}
    bbox_file_temp = os.path.join(task_directory, "bbox.json")
    with open(bbox_file_temp, "w") as f:
        json.dump({"columns": columns_bbox, "rows": rows_bbox}, f, indent=4)

    # read all images, save them in image_1, image_2, ... as PIL images
    images = [ex["figure_path"]]
    image_reading_codes = python_codes_for_table_bbox_load(images, bbox_file_temp)
    image_loading_result = executor.execute(image_reading_codes)
    if image_loading_result[0] != 0:
        raise Exception(f"Error loading images and bounding boxes: {image_loading_result[1]}")

    user = SketchpadUserAgent(
        name="multimodal_user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=checks_terminate_message,
        prompt_generator = prompt_generator,
        parser = parser,
        executor = executor
    )
    
    # running the planning experiment
    all_messages = {}
    
    planner = MultimodalConversableAgent(
        name="planner",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg = lambda x: False,
        system_message=system_prompt,
        llm_config=llm_config
    )
    
    # run the agent
    try:
        user.initiate_chat(
            planner,
            n_image=len(images),
            task_id = "testing_case",
            ex = ex,
            log_prompt_only = False,
        )
        all_messages = planner.chat_messages[user]
        
    except Exception as e:
        print(e)
        all_messages = {'error': e.message if hasattr(e, 'message') else f"{e}"}

    # save the results
    with open(os.path.join(task_directory, "output.json"), "w") as f:
        json.dump(all_messages, f, indent=4, default=custom_encoder)

    # evaluate single accuracy
    model_answer = all_messages[-1]['content'][-1]['text']
    acc_one, prediction, solution = compute_acc_single(ex, model_answer)
    
    with open(os.path.join(task_directory, "answer.json"), "w") as f:
        json.dump({"answer": model_answer, "acc": acc_one, "solution": solution, "prediction": prediction}, f, indent=4)

    user.executor.cleanup()        
    user.reset()
    planner.reset()
    return acc_one

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False, default="tablevqa_wbb", choices=['tablevqa_wbb', 'tablevqa-vtabfact_wbb', 'table-vwtq_syn_wbb'], help='specific tableVQA task')
    parser.add_argument('--output_dir', type=str, required=False, default="mix_col_row_cot", help='edit tool')
    parser.add_argument('--overwrite', type=int, required=False, default=0, help='Overwrite the existing results')
    parser.add_argument('--model_version', type=str, default='gpt-4o-2024-08-06', choices=['gpt-4o-2024-05-13', 'gpt-4o-2024-08-06'])
    args = parser.parse_args()

    input_path = f'data/{args.data}.json'
    print(f"Reading data from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    
    llm_config={"cache_seed": None, "config_list": [{"model": args.model_version, "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}
    args.model_name = args.model_version
    
    # output file
    output_dir_root = 'results'
    output_path = os.path.join(output_dir_root, f'{args.data}/{args.model_name}-{args.output_dir}')
    os.makedirs(output_path, exist_ok=True)

    all_acc = []
    prompt_generator = TablePrompt_mix_col_row_cot()

    for id, ex in tqdm(data.items()):
        output_path_id = f"{output_path}/{id}"
        Path(output_path_id).mkdir(parents=True, exist_ok=True)

        try:
            acc = run_agent(output_path_id, ex, prompt_generator, MAX_REPLY=6, system_prompt=MULTIMODAL_ASSISTANT_MESSAGE)
        except Exception as e:
            acc = 0
            print(e)
        all_acc.append(acc)
        print("acc: ", acc, sum(all_acc) / len(all_acc))
        
    print("acc for tablevqa: ", sum(all_acc) / len(all_acc), len(all_acc))
    print('outputs are saved to', output_path)
