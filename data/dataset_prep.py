# data/dataset_prep.py（Python 3.13适配，高阶版）
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config.config import DATASET_DIR, AUG_CONFIG


def load_yolo_label(label_path):
    """加载YOLO格式标注（适配Python 3.13）"""
    if not os.path.exists(label_path):
        return np.array([])
    try:
        label = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
        return label
    except Exception as e:
        print(f"加载标注失败：{label_path}，错误：{e}")
        return np.array([])


def save_yolo_label(label_path, label):
    """保存YOLO格式标注（适配Python 3.13）"""
    if len(label) == 0:
        return
    np.savetxt(label_path, label, fmt="%.6f")


def split_dataset(test_size=0.2, val_size=0.5):
    """高阶数据集划分（带进度条+异常处理）"""
    img_dir = os.path.join(DATASET_DIR, "images")
    label_dir = os.path.join(DATASET_DIR, "labels")

    # 校验数据集
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        raise ValueError("数据集目录不存在！请确认dataset/images和dataset/labels已创建")

    # 获取图像文件
    img_ext = (".jpg", ".png", ".bmp", ".jpeg")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_ext)]
    if len(img_files) == 0:
        raise ValueError("图像目录为空！")

    # 创建划分目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

    # 划分数据集
    train_imgs, temp_imgs = train_test_split(img_files, test_size=test_size, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_size, random_state=42)

    # 复制文件（带进度条）
    def copy_files(imgs, split):
        for img in tqdm(imgs, desc=f"复制{split}集"):
            # 复制图像
            src_img = os.path.join(img_dir, img)
            dst_img = os.path.join(DATASET_DIR, split, "images", img)
            shutil.copy(src_img, dst_img)

            # 复制标注
            label_name = os.path.splitext(img)[0] + ".txt"
            src_label = os.path.join(label_dir, label_name)
            dst_label = os.path.join(DATASET_DIR, split, "labels", label_name)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    # 生成数据集统计
    stats = {
        "总图像数": len(img_files),
        "训练集": len(train_imgs),
        "验证集": len(val_imgs),
        "测试集": len(test_imgs)
    }
    print("=" * 50)
    print("数据集划分完成（Python 3.13高阶版）：")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("=" * 50)
    return stats


def create_dataset_yaml():
    """生成数据集配置文件（适配Python 3.13）"""
    yaml_path = os.path.join(os.path.dirname(DATASET_DIR), "dataset.yaml")
    yaml_content = f"""# 数据集配置（Python 3.13高阶版）
path: {DATASET_DIR}
train: train/images
val: val/images
test: test/images

# 类别
names:
  0: deer
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"数据集配置文件已生成：{yaml_path}")


if __name__ == "__main__":
    # 执行数据集划分
    split_dataset()
    # 生成配置文件
    create_dataset_yaml()