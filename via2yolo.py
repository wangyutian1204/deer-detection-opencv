import json
import os
from PIL import Image

# ---------------------- 配置路径 ----------------------
VIA_JSON_PATH = "D:/作业/机器学习/大作业/软件/wildlife_monitor_313/dataset/via_project_27Dec2025_11h16m.json"
IMAGE_FOLDER = "D:/作业/机器学习/大作业/软件/wildlife_monitor_313/dataset/images"
CLASS_NAME = "deer"
CLASS_ID = 0  # YOLO类别ID从0开始
# ------------------------------------------------------

print("=== 开始从VIA项目文件生成YOLO标注 ===")

# 读取VIA项目文件
with open(VIA_JSON_PATH, 'r', encoding='utf-8') as f:
    via_data = json.load(f)

# 遍历每张图片的标注
for raw_img_name, img_meta in via_data["_via_img_metadata"].items():
    # 清理VIA里的文件名：去掉.jpg后面的多余数字（比如deer_01.jpg218016 → deer_01.jpg）
    if ".jpg" in raw_img_name:
        img_name = raw_img_name.split(".jpg")[0] + ".jpg"
    else:
        img_name = raw_img_name
        print(f"⚠️ 文件名不是.jpg格式：{raw_img_name}，跳过")
        continue

    # 检查图片是否存在
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    if not os.path.exists(img_path):
        print(f"⚠️ 图片 {img_name} 不存在，跳过")
        continue

    # 获取图片实际尺寸
    with Image.open(img_path) as img:
        img_width, img_height = img.size

    # 生成YOLO的txt文件
    txt_path = os.path.join(IMAGE_FOLDER, os.path.splitext(img_name)[0] + ".txt")
    with open(txt_path, 'w', encoding='utf-8') as txt_f:
        # 遍历该图片的所有标注框
        for region in img_meta["regions"]:
            # 提取矩形标注框坐标
            shape_attr = region["shape_attributes"]
            if shape_attr["name"] != "rect":
                print(f"⚠️ {img_name} 存在非矩形标注，跳过该标注")
                continue
            x = shape_attr["x"]
            y = shape_attr["y"]
            w = shape_attr["width"]
            h = shape_attr["height"]

            # 转换为YOLO归一化格式
            cx = (x + w/2) / img_width
            cy = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            # 写入txt
            txt_f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    print(f"✅ 生成 {img_name} 对应的标注文件 {os.path.basename(txt_path)}")

print("=== 转换完成 ===")