import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTextEdit, QProgressBar, QTabWidget, QGroupBox,
                             QGridLayout, QApplication)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from ultralytics import YOLO

# 内置配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "processed_images"), exist_ok=True)

REPORT_CONFIG = {
    "excel_name": "鹿群计数报告.xlsx",
    "pdf_name": "鹿群计数报告.pdf"
}


# ===================== 照片处理子线程（避免主线程阻塞） =====================
class PhotoProcessThread(QThread):
    result_signal = pyqtSignal(np.ndarray, int)  # 处理后帧+计数
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path
        # 改用轻量模型，降低运算量
        self.model = YOLO("yolov8s-seg.pt")
        self.DEER_CLASS_ID = 49

    def run(self):
        try:
            self.log_signal.emit(f"[{datetime.now()}] 开始处理照片...")
            img = cv2.imread(self.img_path)
            if img is None:
                self.error_signal.emit("照片读取失败！")
                return

            # 降低推理分辨率，减少运算量
            results = self.model(img, conf=0.4, iou=0.5, classes=[self.DEER_CLASS_ID], imgsz=480)
            count = len(results[0].masks.data) if results[0].masks else 0

            # 画框
            processed_img = img.copy()
            if results[0].masks:
                for mask in results[0].masks.data.cpu().numpy():
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(processed_img, contours, -1, (0, 0, 255), 2)

            self.result_signal.emit(processed_img, count)
            self.log_signal.emit(f"[{datetime.now()}] 照片处理完成，计数：{count}只")
        except Exception as e:
            self.error_signal.emit(f"处理错误：{str(e)}")


# ===================== 实时检测线程（轻量模型） =====================
class RealTimeDetectThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    count_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.cap = None
        self.is_running = False
        self.model = YOLO("yolov8s-seg.pt")  # 轻量模型
        self.DEER_CLASS_ID = 49
        self.current_deer_count = 0

    def run(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.log_signal.emit("无法打开摄像头/视频！")
            return

        self.log_signal.emit(f"启动{'摄像头' if self.source == 0 else '视频'}检测...")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.log_signal.emit("检测源断开！")
                break

            # 降低推理分辨率
            results = self.model(frame, conf=0.4, iou=0.5, classes=[self.DEER_CLASS_ID], imgsz=480)
            masks = results[0].masks
            self.current_deer_count = 0
            processed_frame = frame.copy()

            if masks is not None:
                for mask in masks.data.cpu().numpy():
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(processed_frame, contours, -1, (0, 0, 255), 2)
                        self.current_deer_count += 1

            self.frame_signal.emit(processed_frame)
            self.count_signal.emit(self.current_deer_count)

        self.cap.release()
        self.log_signal.emit("检测已停止！")

    def stop(self):
        self.is_running = False


# ===================== 批量处理线程（轻量模型） =====================
class BatchProcessThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)
    log = pyqtSignal(str)

    def __init__(self, img_paths, model):
        super().__init__()
        self.img_paths = img_paths
        self.model = model
        self.DEER_CLASS_ID = 49

    def run(self):
        results = []
        total = len(self.img_paths)
        self.log.emit(f"批量处理{total}张图像（仅检测鹿）...")

        for i, img_path in enumerate(self.img_paths):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.log.emit(f"跳过无效图：{img_path}")
                    continue

                results_model = self.model(img, conf=0.4, classes=[self.DEER_CLASS_ID], imgsz=480)
                masks = results_model[0].masks
                count = len(masks.data) if masks else 0

                processed_img = img.copy()
                if masks:
                    for mask in masks.data.cpu().numpy():
                        mask = (mask > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(processed_img, contours, -1, (0, 0, 255), 2)

                save_path = os.path.join(REPORTS_DIR, "processed_images", os.path.basename(img_path))
                cv2.imwrite(save_path, processed_img)

                results.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_path, count, save_path])
                self.progress.emit(int((i + 1) / total * 100))
                self.log.emit(f"处理完成：{img_path} → 鹿数量：{count}只")
            except Exception as e:
                self.log.emit(f"处理失败：{img_path}，错误：{str(e)}")

        self.result.emit(results)
        self.log.emit(f"批量完成！有效图：{len(results)}张")


# ===================== 主界面（无阻塞+轻量模型） =====================
class WildlifeMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鹿群动态识别计数系统")
        self.setGeometry(100, 100, 1600, 900)

        self.input_path = ""
        self.processed_img = None
        self.count_result = 0
        self.batch_results = []
        self.real_time_thread = None
        self.photo_thread = None  # 照片处理线程
        self.last_frame = None

    def init_ui(self):
        font = QFont("微软雅黑", 10)
        self.setFont(font)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 功能按钮
        btn_layout = QHBoxLayout()
        self.camera_btn = QPushButton("启动摄像头检测")
        self.camera_btn.clicked.connect(self.start_camera_detect)
        self.video_btn = QPushButton("打开鹿群视频")
        self.video_btn.clicked.connect(self.start_video_detect)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_detect)
        self.stop_btn.setEnabled(False)
        self.upload_btn = QPushButton("上传鹿群照片")
        self.upload_btn.clicked.connect(self.upload_deer_img)
        self.process_btn = QPushButton("处理照片")
        self.process_btn.clicked.connect(self.start_photo_process)
        self.process_btn.setEnabled(False)  # 未上传时禁用
        self.export_btn = QPushButton("导出报告")
        self.export_btn.clicked.connect(self.export_report)

        btn_list = [self.camera_btn, self.video_btn, self.stop_btn,
                    self.upload_btn, self.process_btn, self.export_btn]
        for btn in btn_list:
            btn.setMinimumHeight(40)
            btn_layout.addWidget(btn)
        main_layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 核心标签页
        tab_widget = QTabWidget()

        # 实时检测页
        real_time_tab = QWidget()
        rt_layout = QVBoxLayout(real_time_tab)
        self.rt_frame_group = QGroupBox("实时检测画面（鹿会标红框）")
        frame_layout = QVBoxLayout(self.rt_frame_group)
        self.rt_frame_label = QLabel("请启动检测（展示鹿的画面）")
        self.rt_frame_label.setAlignment(Qt.AlignCenter)
        self.rt_frame_label.setStyleSheet("border: 2px solid #0088ff;")
        self.rt_frame_label.setMinimumSize(800, 600)
        frame_layout.addWidget(self.rt_frame_label)
        rt_layout.addWidget(self.rt_frame_group, stretch=1)

        count_layout = QHBoxLayout()
        self.current_count_label = QLabel("当前检测到鹿的数量：0 只")
        self.current_count_label.setStyleSheet("font-size: 16px; color: #ff0000;")
        self.status_label = QLabel("状态：未检测")
        self.status_label.setStyleSheet("font-size: 16px; color: #008800;")
        count_layout.addWidget(self.current_count_label)
        count_layout.addWidget(self.status_label)
        rt_layout.addLayout(count_layout)
        tab_widget.addTab(real_time_tab, "实时动态检测")

        # 照片处理页
        img_tab = QWidget()
        img_layout = QHBoxLayout(img_tab)
        self.origin_img_group = QGroupBox("原始鹿群照片")
        origin_layout = QVBoxLayout(self.origin_img_group)
        self.origin_img_label = QLabel("请上传鹿的照片")
        self.origin_img_label.setAlignment(Qt.AlignCenter)
        self.origin_img_label.setStyleSheet("border: 1px solid #ccc;")
        origin_layout.addWidget(self.origin_img_label)
        img_layout.addWidget(self.origin_img_group, stretch=1)

        self.processed_img_group = QGroupBox("处理后照片（计数结果）")
        processed_layout = QVBoxLayout(self.processed_img_group)
        self.processed_img_label = QLabel("未处理")
        self.processed_img_label.setAlignment(Qt.AlignCenter)
        self.processed_img_label.setStyleSheet("border: 1px solid #ccc;")
        processed_layout.addWidget(self.processed_img_label)
        img_layout.addWidget(self.processed_img_group, stretch=1)
        tab_widget.addTab(img_tab, "鹿群照片处理")

        # 日志页
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        tab_widget.addTab(log_tab, "系统日志")

        main_layout.addWidget(tab_widget, stretch=1)

    # 实时检测功能
    def start_camera_detect(self):
        self.stop_detect()
        self.real_time_thread = RealTimeDetectThread(source=0)
        self._bind_rt_signals()
        self.real_time_thread.start()
        self.camera_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("状态：摄像头检测中（请展示鹿的画面）")

    def start_video_detect(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择鹿群视频", "", "视频文件 (*.mp4 *.avi)"
        )
        if not video_path:
            return
        self.stop_detect()
        self.real_time_thread = RealTimeDetectThread(source=video_path)
        self._bind_rt_signals()
        self.real_time_thread.start()
        self.camera_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("状态：视频检测中（鹿会标红框）")

    def stop_detect(self):
        if self.real_time_thread and self.real_time_thread.isRunning():
            self.real_time_thread.stop()
            self.real_time_thread.wait()
        self.camera_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.last_frame is not None:
            self.update_rt_frame(self.last_frame)
        self.status_label.setText(
            f"状态：检测停止（最后鹿数量：{self.real_time_thread.current_deer_count if self.real_time_thread else 0}只）")

    def _bind_rt_signals(self):
        self.real_time_thread.frame_signal.connect(self.update_rt_frame)
        self.real_time_thread.count_signal.connect(self.update_count)
        self.real_time_thread.log_signal.connect(self.log_text.append)

    def update_rt_frame(self, frame):
        self.last_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        target_w = self.rt_frame_label.width()
        target_h = self.rt_frame_label.height()
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(rgb_frame, (new_w, new_h), cv2.INTER_AREA)
        qimg = QImage(resized_frame.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.rt_frame_label.setPixmap(QPixmap.fromImage(qimg))

    def update_count(self, count):
        self.current_count_label.setText(f"当前检测到鹿的数量：{count} 只")

    # 照片处理功能（子线程版，无阻塞）
    def upload_deer_img(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择鹿群照片", "", "图片文件 (*.jpg *.png)"
        )
        if not file_path:
            return
        self.input_path = file_path
        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]
        scale = min(self.origin_img_label.width() / w, self.origin_img_label.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(rgb_img, (new_w, new_h), cv2.INTER_AREA)
        qimg = QImage(resized_img.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.origin_img_label.setPixmap(QPixmap.fromImage(qimg))
        self.log_text.append(f"[{datetime.now()}] 已上传鹿群照片：{file_path}")
        self.process_btn.setEnabled(True)  # 上传后启用处理按钮

    def start_photo_process(self):
        if not self.input_path:
            QMessageBox.warning(self, "警告", "请先上传鹿群照片！")
            return
        # 启动照片处理子线程
        self.photo_thread = PhotoProcessThread(self.input_path)
        self.photo_thread.result_signal.connect(self.on_photo_process_finish)
        self.photo_thread.log_signal.connect(self.log_text.append)
        self.photo_thread.error_signal.connect(self.on_photo_process_error)
        self.photo_thread.start()
        self.process_btn.setEnabled(False)  # 处理中禁用按钮
        self.log_text.append(f"[{datetime.now()}] 照片处理线程已启动...")

    def on_photo_process_finish(self, processed_img, count):
        # 显示处理后照片
        rgb_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_processed.shape[:2]
        scale = min(self.processed_img_label.width() / w, self.processed_img_label.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_processed = cv2.resize(rgb_processed, (new_w, new_h), cv2.INTER_AREA)
        qimg_processed = QImage(resized_processed.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.processed_img_label.setPixmap(QPixmap.fromImage(qimg_processed))
        self.processed_img_group.setTitle(f"处理后照片（鹿数量：{count}只）")
        self.count_result = count
        self.process_btn.setEnabled(True)  # 处理完成后启用按钮

    def on_photo_process_error(self, error_msg):
        QMessageBox.critical(self, "处理失败", error_msg)
        self.log_text.append(f"[{datetime.now()}] 照片处理失败：{error_msg}")
        self.process_btn.setEnabled(True)

    # 导出报告
    def export_report(self):
        data = []
        has_data = False

        if hasattr(self, 'real_time_thread') and self.real_time_thread is not None:
            if self.real_time_thread.current_deer_count > 0:
                data.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "实时检测（摄像头/视频）",
                    self.real_time_thread.current_deer_count,
                    "动态计数结果"
                ])
                has_data = True

        if self.input_path != "" and self.count_result > 0:
            data.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "静态照片处理",
                self.count_result,
                self.input_path
            ])
            has_data = True

        if len(self.batch_results) > 0:
            data.extend(self.batch_results)
            has_data = True

        if not has_data:
            QMessageBox.warning(self, "警告", "暂无检测数据，请先检测鹿群画面/照片！")
            return

        try:
            from reports.excel_export import export_excel
            excel_path = os.path.join(REPORTS_DIR, REPORT_CONFIG["excel_name"])
            export_excel(data, excel_path)
            self.log_text.append(f"[{datetime.now()}] 计数报告已导出至：{excel_path}")
            QMessageBox.information(self, "成功", f"报告导出成功！\n路径：{excel_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"报告导出错误：{str(e)}")


def run_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    ui = WildlifeMonitorUI()
    ui.init_ui()
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()