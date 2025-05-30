from io import BytesIO
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
import numpy as np

from examples.reward_function.refocus import compute_score

dataset = load_dataset("parquet", data_files={'train': '../train.parquet', 'test': '../test.parquet'})

val_dataset = dataset['test']

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

llm = LLM(model_name, limit_mm_per_prompt={"image": 1})
processor = AutoProcessor.from_pretrained(model_name)

sampling_params = SamplingParams(temperature=0.5, top_p=0.99, max_tokens=1024)

dataloader = DataLoader(val_dataset.with_format("torch"), num_workers=1, batch_size = 1)

print(f"Length of val dataset: {len(dataloader)}")
print(f"Batch size {dataloader.batch_size}")

with open("prompt.txt", 'r') as file:
    template_prompt = file.read()

print(template_prompt)

predicts = []
ground_truths = []

for entry in tqdm(dataloader):

    img_bytes = entry["images"][0][0]

    image = Image.open(BytesIO(img_bytes))

    ground_truths.append(entry["answer"][0])

    #print(entry["prompt"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image },
                {"type": "text", "text": entry["prompt"][0] + template_prompt },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    image_inputs, _ = process_vision_info(messages)
    # Prepare multi-modal data
    multi_modal_data = {"image": image_inputs}

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": multi_modal_data},
        sampling_params=sampling_params,
    )

    output_text = outputs[0].outputs[0].text
    print("[OUTPUT]")
    print(output_text)

    print("[GROUND TRUTH]")
    print(entry["answer"][0])

    predicts.append(output_text)

    #break

scores = compute_score(predicts, ground_truths)
overall_scores = np.array([score["overall"] for score in scores])
overall_mean = np.mean(overall_scores)

print("Mean Overall Score:", overall_mean)