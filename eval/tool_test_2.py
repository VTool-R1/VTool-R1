from PIL import Image
import cv2, json
import numpy as np
from PIL import Image, ImageDraw

from verl.tooluse.tools import *
import json


with open('../data/tablevqa_wbb.json') as json_data:
    d = json.load(json_data)

entry = "nu-3655"
print(d[entry])

record = d[entry]

img_path = record["figure_path"]
img = Image.open("../" + img_path)

c_box = dict(zip(record["column_headers"], record["columns_bbox"]))
r_box = dict(zip(record["row_starters"], record["rows_bbox"]))

image_with_focused_columns = focus_on_columns_with_draw(img, ["Started"], c_box)
image_with_focused_rows = focus_on_rows_with_draw(image_with_focused_columns, ["Kenneth W. Dam", "Alan C. Kohn"], r_box)

image_with_focused_rows.save("test.png")