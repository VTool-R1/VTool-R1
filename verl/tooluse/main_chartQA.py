import json
import os
import argparse
from pathlib import Path
from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt_need import ChartQAPrompt_mix_orig, python_codes_for_chart_bbox_load, MULTIMODAL_ASSISTANT_MESSAGE
from parse import Parser
from execution import CodeExecutor
from utils import custom_encoder
from chart_data import compute_acc_from_raw_answer
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

    x_values = ex['x_values']
    y_values = ex['y_values']
    x_bboxes = ex['x_bboxes']
    y_bboxes = ex['y_bboxes']
    x_values_bbox = {x: x_bboxes[i] for i, x in enumerate(x_values)} if x_values else {}
    y_values_bbox = {y: y_bboxes[i] for i, y in enumerate(y_values)} if y_values else {}
    bbox_file_temp = os.path.join(task_directory, "bbox.json")
    with open(bbox_file_temp, "w") as f:
        json.dump({"x_values_bbox": x_values_bbox, "y_values_bbox": y_values_bbox}, f, indent=4)

    images = [ex["figure_path"]]
    image_reading_codes = python_codes_for_chart_bbox_load(images, bbox_file_temp)
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
            n_image=0,
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

    model_answer = all_messages[-1]['content'][-1]['text']
    solution = ex['answer']
    acc_one, prediction = compute_acc_from_raw_answer(ex['query'], ex['answer'], model_answer)

    with open(os.path.join(task_directory, "answer.json"), "w") as f:
        json.dump({"answer": model_answer, "acc": acc_one, "solution": solution, "prediction": prediction}, f, indent=4)
        
    user.executor.cleanup()
    user.reset()
    planner.reset()
    return acc_one



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False, default="chartQA_h_bar_wbb", choices=['chartQA_h_bar_wbb', 'chartQA_v_bar_wbb'], help='specific chartQA task')
    parser.add_argument('--output_dir', type=str, required=False, default="mix_orig")
    parser.add_argument('--overwrite', type=int, required=False, default=0)
    parser.add_argument('--model_version', type=str, default='gpt-4o-2024-08-06', choices=['gpt-4o-2024-05-13', 'gpt-4o-2024-08-06'])
    args = parser.parse_args()

    input_path = f'data/{args.data}.json'
    print(f"Reading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    llm_config={"cache_seed": None, "config_list": [{"model": args.model_version, "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}
    args.model_name = args.model_version

    # output file
    output_dir_root = 'results'
    output_path = os.path.join(output_dir_root, f'{args.data}/{args.model_name}-{args.output_dir}')
    print("Output file:", output_path)
    os.makedirs(output_path, exist_ok=True)


    all_acc = []
    prompt_generator = ChartQAPrompt_mix_orig()

    for id, ex in tqdm(data.items()):
        output_path_id = f"{output_path}/{id}"
        Path(output_path_id).mkdir(parents=True, exist_ok=True)
        
        try:
            acc = run_agent(output_path_id, ex, prompt_generator, MAX_REPLY=6, system_prompt=MULTIMODAL_ASSISTANT_MESSAGE)
        except Exception as e:
            acc = 0
            print(e)
        all_acc.append(acc)
        print("acc: ", sum(all_acc) / len(all_acc))

    print("acc for chartQA: ", sum(all_acc) / len(all_acc))
    print('outputs are saved to', output_path)
