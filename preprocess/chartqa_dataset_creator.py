import json
from datasets import Dataset, DatasetDict
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

split = "val"

with open(f'data/chartqa_vcot/{split}.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

figure_id = []
figure_path = []
images = []
#annotation_path = []
#table_path = []
query = []
prompt = []
answer = []
metadata = []

# Extract data
for record in tqdm(data):

    t = record["source"]

    '''if t != 'v_bar':
        continue'''

    '''if t != "v_bar" and t != "h_bar":
        print(t)
        print("type not valid")'''

    figure_id.append(record["id"])
    #annotation_path.append(record.get("annotation_path"))
    #table_path.append(record.get("table_path"))

    full_prompt = f"""<image> # USER REQUEST #: {record.get("query")}
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: {list(record["x_values_bbox"].keys())}. The y values in the image are: {list(record["y_values_bbox"].keys())}.
# USER IMAGE stored in image_1, as PIL image."""

    query.append(record.get("query"))
    prompt.append(full_prompt)

    #print(full_prompt)

    ans = record.get("answer")
    if isinstance(ans, list):
        ans = "|||".join([str(a) for a in ans])

    answer.append(ans)

    img_path = record["conversations"][0]["images"][0]
    img_path = img_path.replace("ChartQAtest", "ChartQA/test").replace("ChartQAval", "ChartQA/val")

    figure_path.append(img_path)

    '''metadata.append({
        "type": record.get("type"),
        "figure_bbox": record.get("figure_bbox"),
        "x_values": record.get("x_values"),
        "y_values": record.get("y_values"),
        "x_bboxes": record.get("x_bboxes"),
        "y_bboxes": record.get("y_bboxes"),
    })'''

    metadata.append({
        "type": record["source"],
        "figure_bbox": record["figure_bbox"],
        "x_values_bbox": record["x_values_bbox"] or {"x1": "none"},
        "y_values_bbox": record["y_values_bbox"] or {"x1": "none"},
    })

    with Image.open(img_path) as img:
        buffer = BytesIO()
        img.save(buffer, format=img.format)  # Keep the same format (e.g., JPEG, PNG)
        image_bytes = buffer.getvalue()
       #print(type(image_bytes))
        images.append([image_bytes])

    #quit()

merged_dict = {
    "metadata": metadata,
    "figure_id": figure_id,
    "figure_path": figure_path,
    #"annotation_path": annotation_path,
    #"table_path": table_path,
    "query": query,
    "prompt": prompt,
    "answer": answer,
    "images": images
}

print("TOTAL SIZE: ", len(metadata))

ds = Dataset.from_dict(merged_dict)

ds.to_parquet(f"{split}_full.parquet")

'''
split_ds = ds.train_test_split(test_size=0.1, seed=492)

train_dataset = split_ds['train']
test_dataset = split_ds['test']

# Save as parquet files
train_dataset.to_parquet("train.parquet")
test_dataset.to_parquet("test.parquet")'''