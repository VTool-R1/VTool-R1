from io import BytesIO
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

from examples.reward_function.refocus import compute_score

dataset = load_dataset("parquet", data_files={'train': '../train.parquet', 'test': '../test.parquet'})

val_dataset = dataset['test']

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

dataloader = DataLoader(val_dataset.with_format("torch"), num_workers=1)

with open("prompt.txt", 'r') as file:
    template_prompt = file.read()

predicts = []
ground_truths = []

for entry in tqdm(dataloader):

    img_bytes= entry["images"][0][0]

    image = Image.open(BytesIO(img_bytes))

    ground_truths.append(entry["answer"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image },
                {"type": "text", "text": entry["prompt"] + template_prompt },
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
    print(output_text)

    predicts.append(output_text)

    break

rewards = compute_score(predicts)