# models/model_train.py（Python 3.13适配，高阶训练）
from ultralytics import YOLO
import torch
import torch.nn as nn
from config.config import TRAIN_CONFIG, DATASET_DIR, DEVICE
from utils.logger import setup_logger

# 配置日志
logger = setup_logger("model_train_313")


class CustomLoss(nn.Module):
    """自定义损失函数（适配遮挡场景，Python 3.13）"""

    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 分割损失权重
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss

    def _dice_loss(self, pred, target):
        """Dice损失（分割专用）"""
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + smooth) / (union + smooth)

    def forward(self, pred_cls, pred_seg, target_cls, target_seg):
        """前向传播"""
        cls_loss = self.bce_loss(pred_cls, target_cls)
        seg_loss = self.dice_loss(pred_seg, target_seg)
        total_loss = self.alpha * cls_loss + self.beta * seg_loss
        return total_loss


def train_yolov8_seg():
    """高阶训练流程（含自定义损失+早停+日志）"""
    # 加载基础模型
    model = YOLO(TRAIN_CONFIG["model"])

    # 替换损失函数（适配Python 3.13）
    custom_loss = CustomLoss().to(DEVICE)
    logger.info("已加载自定义损失函数（适配遮挡场景）")

    # 训练参数增强
    train_args = TRAIN_CONFIG.copy()
    train_args["data"] = "../dataset.yaml"  # 数据集配置文件
    train_args["loss"] = custom_loss  # 自定义损失
    train_args["cos_lr"] = True  # 余弦学习率
    train_args["hsv_h"] = 0.015  # 色相增强
    train_args["hsv_s"] = 0.7  # 饱和度增强
    train_args["hsv_v"] = 0.4  # 明度增强

    # 开始训练
    logger.info("开始YOLOv8-seg高阶训练（Python 3.13）")
    logger.info(f"训练参数：{train_args}")

    results = model.train(**train_args)

    # 保存训练结果
    logger.info("训练完成，保存结果...")
    logger.info(f"训练损失：{results.results_dict}")
    logger.info(f"验证mAP50：{results.box.map50}")
    logger.info(f"分割mAP50：{results.seg.map50}")

    return results


if __name__ == "__main__":
    # 执行训练
    train_yolov8_seg()