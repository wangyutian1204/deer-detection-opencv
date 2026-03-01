# reports/pdf_export.py（完整版，适配Python 3.13）
import os
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

def export_pdf(results, save_path, processed_img=None, is_batch=False):
    """导出PDF报告（适配单张/批量处理）"""
    # 创建PDF画布
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()

    # 标题
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-50, "鹿群识别与计数报告（Python 3.13）")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height-70, f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 内容区域
    y_pos = height - 100

    # 1. 单张处理：添加分割后图像
    if not is_batch and processed_img is not None:
        # 保存临时图像
        temp_img = os.path.join(os.path.dirname(save_path), "temp_pdf_img.png")
        cv2.imwrite(temp_img, processed_img)
        # 绘制图像
        img = ImageReader(temp_img)
        c.drawImage(img, width/2 - 150, y_pos - 200, width=300, height=200)
        c.drawString(width/2 - 150, y_pos - 210, "分割后图像（含计数标注）")
        os.remove(temp_img)
        y_pos -= 220

    # 2. 计数结果表格
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "计数结果统计")
    c.setFont("Helvetica", 12)
    y_pos -= 20

    # 表格数据
    headers = ["处理时间", "文件路径", "计数结果（只）"]
    if len(results[0]) == 4:
        headers.append("处理后路径")
    table_data = [headers] + results

    # 绘制表格
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    table.wrapOn(c, width-100, height)
    table.drawOn(c, 50, y_pos - len(table_data)*20)
    y_pos -= len(table_data)*20 + 20

    # 3. 统计信息（批量处理）
    if is_batch:
        counts = [row[2] for row in results]
        total = sum(counts)
        avg = total / len(counts)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, f"统计总结：总处理{len(results)}张，总计数{total}只，平均每图{avg:.2f}只")
        y_pos -= 20

    # 4. 页脚
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "基于YOLOv8-seg的无人机鹿群识别系统（Python 3.13适配）")
    c.drawString(50, 30, "版本：V1.0（高阶版）")

    # 保存PDF
    c.save()
    print(f"PDF报告已导出：{save_path}")