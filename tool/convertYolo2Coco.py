LABEL_DIR = "/Users/nakamura/git/kunshujo/omekac_kunshujo/docs/files/labels"
IMAGE_DIR = "/Users/nakamura/git/kunshujo/omekac_kunshujo/docs/files/yolov5"
YAML_PATH = "data.yaml"
OUTPUT_IMAGE_DIR="img"
OUTPUT_LABEL_PATH = "dataset.json"

# 設定終了

import yaml
import pprint
import glob
import os
from pathlib import Path
from PIL import Image
import json
import shutil
from tqdm import tqdm

## 出力画像フォルダの作成
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

categories = []

with open(YAML_PATH) as file:
    obj = yaml.safe_load(file)
    names = obj["names"]

    for i in range(len(names)):
        categories.append({"id": i, "name": names[i], "supercategory": "none"})

    # pprint.pprint(names)

files_img = glob.glob(f"{IMAGE_DIR}/*.jpg")

files_img.sort()

images = []
annotations = []

for path_img in tqdm(files_img):
    # print(path_img)

    img = Image.open(path_img)
    w_img, h_img = img.size

    basename_img = os.path.basename(path_img)

    # print(basename_img)

    id_img = Path(path_img).stem

    images.append({
        "file_name": basename_img,
        "width": w_img,
        "height": h_img,
        "id": id_img
    })

    # 画像のコピー

    copy_path_img = f"{OUTPUT_IMAGE_DIR}/{basename_img}"
    if not os.path.exists(copy_path_img):
        shutil.copy(path_img, copy_path_img)

    # 

    path_label = os.path.join(LABEL_DIR, basename_img.replace(".jpg", ".txt"))
    f = open(path_label, 'r', encoding='UTF-8')
    data_label = f.read().strip().split("\n")
    # print(data_label)
    f.close()

    for line in data_label:
        line = line.split(" ")
        yolo_center_x = float(line[1])
        yolo_center_y = float(line[2])
        yolo_width = float(line[3])
        yolo_height = float(line[4])
        name_index = int(line[0])

        x = int((yolo_center_x - yolo_width / 2) * w_img)
        y = int((yolo_center_y - yolo_height / 2) * h_img)
        # x2 = (center_x + width / 2) * w_img
        # y2 = (center_y + height / 2) * h_img

        # x = int(x1)
        # y = int(y1)
        w = int(yolo_width * w_img) # int(x2 - x1)
        h = int(yolo_height * h_img) # int(y2 - y1)

        annotations.append({
            "area": w * h,
            "iscrowd": 0,
            "image_id": id_img,
            "bbox": [x, y, w, h],
            "category_id": name_index,
            "id": len(annotations),
            "ignore": 0,
            "segmentation": []
        })

    # break

result = {
    "images": images,
    "type": "instances",
    "annotations": annotations,
    "categories": categories
}

with open(OUTPUT_LABEL_PATH, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)