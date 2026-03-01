import json
import os

# ---------------------- 配置你的路径（确认这3行和实际一致） ----------------------
COCO_JSON_PATH = "D:/作业/机器学习/大作业/软件/wildlife_monitor_313/dataset/annotations/instances_train.json"
IMAGE_FOLDER = "D:/作业/机器学习/大作业/软件/wildlife_monitor_313/dataset/images"
CLASS_NAME = "deer"
# -----------------------------------------------------------------------

# 打印日志，排查路径问题
print("=== 开始转换 COCO → YOLO ===")
print(f"COCO文件路径：{COCO_JSON_PATH}")
print(f"图片文件夹路径：{IMAGE_FOLDER}")

# 检查路径是否存在
if not os.path.exists(COCO_JSON_PATH):
    print(f"❌ 错误：COCO标注文件不存在！路径：{COCO_JSON_PATH}")
    exit()
if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ 错误：图片文件夹不存在！路径：{IMAGE_FOLDER}")
    exit()

# 类别映射
class2id = {CLASS_NAME: 0}

# 读取COCO文件
try:
    with open(COCO_JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    print(
        f"✅ 成功读取COCO文件，包含 {len(coco_data.get('images', []))} 张图片，{len(coco_data.get('annotations', []))} 个标注")
except Exception as e:
    print(f"❌ 读取COCO文件失败：{str(e)}")
    exit()

# 建立图片ID到文件名/尺寸的映射
img_id2info = {}
for img in coco_data.get('images', []):
    img_id = img['id']
    img_name = img['file_name']
    img_width = img['width']
    img_height = img['height']
    img_id2info[img_id] = (img_name, img_width, img_height)

print(f"✅ 建立图片映射完成，共映射 {len(img_id2info)} 张图片")

# 生成YOLO txt文件
generated_count = 0
for ann in coco_data.get('annotations', []):
    try:
        img_id = ann['image_id']
        # 检查图片ID是否存在
        if img_id not in img_id2info:
            print(f"⚠️ 警告：标注的图片ID {img_id} 不存在，跳过该标注")
            continue

        img_name, img_width, img_height = img_id2info[img_id]
        # 解析多边形标注，计算边界框
        segmentation = ann['segmentation'][0]  # VIA导出的多边形坐标 [x1,y1,x2,y2,...]
        x_coords = segmentation[::2]  # 取偶数位：x1,x2,x3...
        y_coords = segmentation[1::2]  # 取奇数位：y1,y2,y3...

        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)

        # 转换为YOLO归一化格式（class_id cx cy w h）
        cx = (xmin + xmax) / 2 / img_width
        cy = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        # 生成txt文件路径
        txt_filename = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(IMAGE_FOLDER, txt_filename)

        # 写入txt文件
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f"{class2id[CLASS_NAME]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        generated_count += 1
    except Exception as e:
        print(f"⚠️ 处理标注ID {ann.get('id', '未知')} 失败：{str(e)}")
        continue

print(f"=== 转换完成 ===")
print(f"✅ 成功生成 {generated_count} 个标注行，对应文件在 {IMAGE_FOLDER} 下")
print(f"❓ 若生成数为0：检查COCO文件是否有标注，或图片ID匹配是否正确")