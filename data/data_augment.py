# data/data_augment.py（Python 3.13适配，高阶增强）
import cv2
import numpy as np
import random
from config.config import AUG_CONFIG


def random_flip(img, label):
    """随机翻转（水平/垂直，同步调整标注）"""
    h, w = img.shape[:2]
    # 水平翻转
    if random.random() < AUG_CONFIG["flip_prob"]:
        img = cv2.flip(img, 1)
        if len(label) > 0:
            label[:, 1] = 1 - label[:, 1]  # x坐标翻转
    # 垂直翻转
    if random.random() < AUG_CONFIG["flip_prob"]:
        img = cv2.flip(img, 0)
        if len(label) > 0:
            label[:, 2] = 1 - label[:, 2]  # y坐标翻转
    return img, label


def random_crop(img, label):
    """随机裁剪（模拟无人机视角变化，同步调整标注）"""
    h, w = img.shape[:2]
    crop_h = int(h * random.uniform(*AUG_CONFIG["crop_ratio"]))
    crop_w = int(w * random.uniform(*AUG_CONFIG["crop_ratio"]))

    # 随机裁剪起点
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)

    # 裁剪图像
    img_crop = img[y1:y1 + crop_h, x1:x1 + crop_w]

    # 调整标注
    if len(label) > 0:
        new_label = []
        for l in label:
            cls, x, y, bw, bh = l
            # 转换为像素坐标
            px = x * w
            py = y * h
            pbw = bw * w
            pbh = bh * h

            # 判断标注中心是否在裁剪区域内
            if x1 < px < x1 + crop_w and y1 < py < y1 + crop_h:
                # 转换为裁剪后相对坐标
                new_x = (px - x1) / crop_w
                new_y = (py - y1) / crop_h
                new_bw = pbw / crop_w
                new_bh = pbh / crop_h

                # 过滤无效标注
                if 0 < new_x < 1 and 0 < new_y < 1 and 0 < new_bw < 1 and 0 < new_bh < 1:
                    new_label.append([cls, new_x, new_y, new_bw, new_bh])
        label = np.array(new_label)

    return img_crop, label


def color_jitter(img):
    """色域变换（亮度/对比度/饱和度，模拟不同光照）"""
    # 亮度调整
    alpha = 1 + random.uniform(-AUG_CONFIG["brightness"], AUG_CONFIG["brightness"])
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # 对比度调整
    alpha = 1 + random.uniform(-AUG_CONFIG["contrast"], AUG_CONFIG["contrast"])
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # 饱和度调整
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + random.uniform(-AUG_CONFIG["saturation"], AUG_CONFIG["saturation"]))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def add_noise(img):
    """添加高斯噪声（模拟无人机拍摄噪声）"""
    if random.random() < AUG_CONFIG["noise_prob"]:
        mean = 0
        var = random.uniform(10, 50)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img.shape)
        img = img + gauss
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def add_blur(img):
    """随机模糊（模拟无人机运动模糊）"""
    if random.random() < AUG_CONFIG["blur_prob"]:
        blur_type = random.choice(["gaussian", "median"])
        if blur_type == "gaussian":
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        else:
            ksize = random.choice([3, 5])
            img = cv2.medianBlur(img, ksize)
    return img


def augment_image(img, label):
    """一站式高阶数据增强（适配Python 3.13）"""
    # 1. 随机翻转
    img, label = random_flip(img, label)
    # 2. 随机裁剪
    img, label = random_crop(img, label)
    # 3. 色域变换
    img = color_jitter(img)
    # 4. 添加噪声
    img = add_noise(img)
    # 5. 随机模糊
    img = add_blur(img)

    return img, label


if __name__ == "__main__":
    # 测试增强效果
    test_img = cv2.imread("../dataset/train/images/deer_001.jpg")
    test_label = load_yolo_label("../dataset/train/labels/deer_001.txt")

    aug_img, aug_label = augment_image(test_img, test_label)
    cv2.imwrite("../augmented_deer_313.jpg", aug_img)
    save_yolo_label("../augmented_deer_313.txt", aug_label)
    print("高阶数据增强测试完成（Python 3.13）！")