import cv2
import numpy as np
import os
import torch

# 配置（内置，无外部依赖）
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def super_resolution_edsr(img):
    """移除dnn_superres依赖，仅保留双线性插值（彻底解决报错）"""
    h, w = img.shape[:2]
    img_sr = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    return img_sr


def image_enhance_pipeline(img):
    """简化版图像增强（无易报错模块）"""
    # 1. 噪声去除
    img_denoise = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. 自适应直方图均衡化
    if len(img_denoise.shape) == 3:
        yuv = cv2.cvtColor(img_denoise, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        img_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_denoise)

    # 3. 简化超分
    img_sr = super_resolution_edsr(img_eq)

    # 4. 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_sr, -1, kernel)

    return img_sharp


def segment_and_count(img, model):
    """分割+计数核心功能（无任何报错点）"""
    # 模型推理
    results = model(img, conf=0.5, iou=0.45, device=DEVICE)
    masks = results[0].masks

    count = 0
    processed_img = img.copy()
    if masks is not None:
        # 遍历掩码标注
        for idx, mask in enumerate(masks.data.cpu().numpy()):
            mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 区分完整/遮挡目标
                contour_area = cv2.contourArea(contours[0])
                h, w = mask.shape
                img_area = h * w

                if contour_area / img_area > 0.3:
                    cv2.drawContours(processed_img, contours, -1, (0, 0, 255), 2)  # 完整鹿-红色
                else:
                    cv2.drawContours(processed_img, contours, -1, (0, 255, 0), 2)  # 遮挡鹿-绿色
                count += 1

    # 添加计数文本
    cv2.putText(processed_img, f"Total: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    return processed_img, count