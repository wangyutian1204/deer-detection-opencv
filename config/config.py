import os
import torch

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "processed_images"), exist_ok=True)

# 设备配置
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

# 训练配置
TRAIN_CONFIG = {
    "model": "yolov8s-seg.pt",  # 基础模型
    "epochs": 100,              # 训练轮数（比基础版多）
    "batch_size": 16,           # 批次大小
    "img_size": 640,            # 输入尺寸
    "lr0": 0.01,                # 初始学习率
    "lrf": 0.01,                # 最终学习率
    "weight_decay": 0.0005,     # 权重衰减
    "warmup_epochs": 3,         # 预热轮数
    "patience": 10,             # 早停耐心值
    "save_period": 5,           # 权重保存间隔
    "device": DEVICE,
    "project": RUNS_DIR,
    "name": "deer_seg_313_higher",
    "exist_ok": True,
    "val": True,
    "plots": True
}

# 评估配置
EVAL_CONFIG = {
    "conf": 0.5,                # 置信度阈值
    "iou": 0.45,                # IOU阈值
    "split": "test",            # 测试集评估
    "save_json": True,
    "verbose": True
}

# 数据增强配置
AUG_CONFIG = {
    "flip_prob": 0.5,           # 翻转概率
    "crop_ratio": (0.7, 1.0),   # 裁剪比例
    "brightness": 0.3,          # 亮度调整范围
    "contrast": 0.3,            # 对比度调整范围
    "saturation": 0.3,          # 饱和度调整范围
    "noise_prob": 0.2,          # 噪声添加概率
    "blur_prob": 0.2            # 模糊概率
}

# 超分辨率配置
SR_CONFIG = {
    "model_path": os.path.join(BASE_DIR, "EDSR_x4.pb"),  # 4倍超分模型
    "scale": 4,                 # 放大倍数
    "model_type": "edsr"        # 模型类型
}

# 报告配置
REPORT_CONFIG = {
    "excel_name": "鹿群计数报告_高阶版_313.xlsx",
    "pdf_name": "鹿群计数报告_高阶版_313.pdf",
    "font_path": "simhei.ttf",  # 中文显示字体
    "dpi": 300                  # PDF分辨率
}