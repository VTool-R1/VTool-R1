import json
from datasets import Dataset, DatasetDict
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

def GetImageBytes(image_path):
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format=img.format)  # Keep the same format (e.g., JPEG, PNG)
        image_bytes = buffer.getvalue()
        return image_bytes
    
def get_image_size(image_path):
    """Returns the width and height of an image loaded from disk."""
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

split = "val"

figure_id = []
figure_path = []
images = []
#annotation_path = []
#table_path = []
query = []
prompt = []
answer = []
metadata = []

seen_keys = set()

with open(f'data/tablevqa-vtabfact_wbb.json', 'r') as file:
    data = json.load(file)

print(len(data.keys()))

def AddEntry(record):
    figure_id.append(record["figure_id"])
    query.append(record["query"])

    full_prompt = f"""<image># USER REQUEST #: {record.get("query")}
# USER Bounding Box Info: columns_bbox, where keys are column headers and values are column bounding boxes. rows_bbox, where keys row headers and values are row bounding boxes. The columns in the image are: {record["column_headers"]}. The rows in the image start with: {record["row_starters"]}.
# USER IMAGE stored in image_1, as PIL image."""
    prompt.append(full_prompt)

    ans = record.get("answer")
    if isinstance(ans, list):
        ans = "|||".join([str(a) for a in ans])

    answer.append(ans)

    img_path = record["figure_path"]
    figure_path.append(img_path)
    images.append([GetImageBytes(img_path)])

    width, height = get_image_size(img_path)

    for bbox in record["rows_bbox"]:
        bbox["x1"] *= width
        bbox["x2"] *= width
        bbox["y1"] *= height
        bbox["y2"] *= height

    for bbox in record["columns_bbox"]:
        bbox["x1"] *= width
        bbox["x2"] *= width
        bbox["y1"] *= height
        bbox["y2"] *= height

    #print(record["rows_bbox"])

    c_box = dict(zip(record["column_headers"], record["columns_bbox"]))
    r_box = dict(zip(record["row_starters"], record["rows_bbox"]))

    metadata.append(json.dumps({
        "type": "table",
        "figure_bbox": record["table_bbox"],
        "columns_bbox": c_box,
        "row_starters": r_box,
    }))

    #print(full_prompt)

for key in data.keys():
    if key in seen_keys:
        continue

    seen_keys.add(key)

    record = data[key]
    AddEntry(record)

with open(f'data/tablevqa_wbb.json', 'r') as file:
    data = json.load(file)

print(len(data.keys()))

for key in data.keys():
    if key in seen_keys:
        continue
    seen_keys.add(key)

    record = data[key]
    AddEntry(record)

with open(f'data/tablevqa-vwtq_syn_wbb.json', 'r') as file:
    data = json.load(file)

print(len(data.keys()))

for key in data.keys():
    if key in seen_keys:
        continue
    seen_keys.add(key)

    record = data[key]
    AddEntry(record)


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

ds = Dataset.from_dict(merged_dict)
ds = ds.shuffle(seed=42)

split_ds = ds.train_test_split(test_size=0.3, seed=42)

train_ds = split_ds['train']
test_ds = split_ds['test']

train_ds.to_parquet("table_train.parquet")
test_ds.to_parquet("table_test.parquet")