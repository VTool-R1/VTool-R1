from datasets import load_dataset

from PIL import Image
import json
from verl.tooluse.tools import *

dataset = load_dataset("parquet", data_files={'train': '../datasets/table_train.parquet', 'test': '../datasets/table_test.parquet'})

print(dataset["train"])

entry = dataset["train"][4]


#print(entry["metadata"])
print(entry["prompt"])
print(entry["figure_path"])
#print(entry["figure_path"])

metadata = json.loads(entry["metadata"])

#quit()

#print(metadata["columns_bbox"])
#print(metadata["row_starters"])

img = Image.open("../" + entry["figure_path"])

print(metadata["columns_bbox"]["No. inseason"])

img = focus_on_columns_with_highlight(img, ["No. inseason"], metadata["columns_bbox"])
img.save("table_view.png")

quit()