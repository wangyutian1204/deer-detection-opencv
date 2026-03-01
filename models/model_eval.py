# models/model_eval.py（Python 3.13适配，高阶评估）
from ultralytics import YOLO
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config.config import EVAL_CONFIG, RUNS_DIR, REPORTS_DIR
from utils.logger import setup_logger

# 配置日志和绘图
logger = setup_logger("model_eval_313")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False


def evaluate_model(model_path):
    """高阶模型评估（多维度指标）"""
    # 加载模型
    model = YOLO(model_path)
    logger.info(f"加载模型：{model_path}")

    # 执行评估
    metrics = model.val(**EVAL_CONFIG)

    # 提取详细指标
    eval_results = {
        # 边界框指标
        "box_mAP50": metrics.box.map50,
        "box_mAP95": metrics.box.map95,
        "box_precision": metrics.box.p,
        "box_recall": metrics.box.r,
        "box_f1": metrics.box.f1,
        # 分割指标
        "seg_mAP50": metrics.seg.map50,
        "seg_mAP95": metrics.seg.map95,
        "seg_precision": metrics.seg.p,
        "seg_recall": metrics.seg.r,
        "seg_f1": metrics.seg.f1,
        # 速度指标
        "inference_time": metrics.speed["inference"],
        "preprocess_time": metrics.speed["preprocess"],
        "postprocess_time": metrics.speed["postprocess"]
    }

    # 保存评估结果
    eval_path = os.path.join(REPORTS_DIR, "model_evaluation_313.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    logger.info(f"评估结果已保存：{eval_path}")

    # 绘制评估指标图
    plot_evaluation(eval_results)

    # 打印结果
    logger.info("=" * 60)
    logger.info("高阶模型评估结果（Python 3.13）：")
    for k, v in eval_results.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    return eval_results


def plot_evaluation(results):
    """绘制评估指标可视化图（适配Python 3.13）"""
    # 1. 分割vs边界框指标对比
    metrics = ["mAP50", "precision", "recall", "f1"]
    box_vals = [results[f"box_{m}"] for m in metrics]
    seg_vals = [results[f"seg_{m}"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 指标对比图
    ax1.bar(x - width / 2, box_vals, width, label="边界框")
    ax1.bar(x + width / 2, seg_vals, width, label="分割")
    ax1.set_title("边界框vs分割指标对比（Python 3.13）")
    ax1.set_xlabel("指标")
    ax1.set_ylabel("值")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 速度指标图
    speed_metrics = ["inference_time", "preprocess_time", "postprocess_time"]
    speed_vals = [results[m] for m in speed_metrics]
    ax2.bar(speed_metrics, speed_vals, color=["red", "green", "blue"])
    ax2.set_title("推理速度指标（ms/张）")
    ax2.set_ylabel("时间（ms）")
    ax2.grid(True, alpha=0.3)

    # 保存图片
    plot_path = os.path.join(REPORTS_DIR, "evaluation_plot_313.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"评估可视化图已保存：{plot_path}")


if __name__ == "__main__":
    # 评估训练好的模型
    model_path = os.path.join(RUNS_DIR, "deer_seg_313_higher", "weights", "best.pt")
    evaluate_model(model_path)