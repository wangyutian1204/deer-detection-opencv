import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime


# ===================== 新增：登录窗口类（直接添加在这里） =====================
class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("鹿群识别系统 - 登录")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        self.user_role = ""  # 存储登录角色（admin/user）

        # 用户名标签+输入框
        ttk.Label(root, text="用户名：", font=("微软雅黑", 11)).pack(pady=15)
        self.username_entry = ttk.Entry(root, font=("微软雅黑", 11), width=30)
        self.username_entry.pack(pady=5)
        self.username_entry.insert(0, "admin")  # 默认填充管理员账号

        # 密码标签+输入框（隐藏输入）
        ttk.Label(root, text="密码：", font=("微软雅黑", 11)).pack(pady=10)
        self.password_entry = ttk.Entry(root, font=("微软雅黑", 11), width=30, show="*")
        self.password_entry.pack(pady=5)
        self.password_entry.insert(0, "123456")  # 默认填充密码

        # 角色选择（单选按钮）
        self.role_var = tk.StringVar(value="admin")
        role_frame = ttk.Frame(root)
        role_frame.pack(pady=15)
        ttk.Radiobutton(role_frame, text="管理员（全功能）", variable=self.role_var, value="admin").pack(side=tk.LEFT,
                                                                                                       padx=10)
        ttk.Radiobutton(role_frame, text="普通用户（仅查看）", variable=self.role_var, value="user").pack(side=tk.LEFT,
                                                                                                        padx=10)

        # 登录按钮
        ttk.Button(root, text="登录", command=self.check_login, style="Custom.TButton").pack(pady=10)

    def check_login(self):
        # 简单验证逻辑（无需数据库，适合大作业）
        username = self.username_entry.get()
        password = self.password_entry.get()
        role = self.role_var.get()

        # 验证通过条件
        if (role == "admin" and username == "admin" and password == "123456") or \
                (role == "user" and username == "user" and password == "123456"):
            self.user_role = role
            self.root.destroy()  # 关闭登录窗口，进入主程序
        else:
            messagebox.showerror("登录失败", "用户名/密码/角色不匹配！")


# ===================== 项目路径配置 =====================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
ORIGINAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "original")
MARKED_DIR = os.path.join(PROJECT_ROOT, "dataset", "marked")
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(MARKED_DIR, exist_ok=True)

# 照片计数配置
PHOTO_COUNT_DICT = {
    "deer_01": 80,
    "deer_02": 15,
    "deer_03": 88,
    "deer_04": 67,
    "deer_05": 92,
    "deer_06": 45,
    "deer_07": 71,
    "deer_08": 83,
    "deer_09": 59,
    "deer_10": 68
}

# 统一照片基础尺寸
BASE_PHOTO_WIDTH = 600
BASE_PHOTO_HEIGHT = 400

# ===================== 极简检测参数（不影响播放） =====================
DIFF_THRESHOLD = 20  # 提高灵敏度
CONTOUR_AREA_THRESHOLD = 200  # 小目标也能检测
BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2


# ===================== 界面样式配置 =====================
def init_style():
    style = ttk.Style()
    style.configure(".", font=("微软雅黑", 10))
    style.configure(
        "Custom.TButton",
        font=("微软雅黑", 10, "bold"),
        padding=(12, 6),
        relief="raised"
    )
    style.configure(
        "Custom.TNotebook",
        tabmargins=[10, 10, 5, 5],
        font=("微软雅黑", 11, "bold"),
        background="#f5f5f5"
    )
    style.configure(
        "Title.TLabel",
        font=("微软雅黑", 12, "bold"),
        foreground="#2c3e50",
        background="#f5f5f5"
    )
    style.configure(
        "Count.TLabel",
        font=("微软雅黑", 11, "bold"),
        foreground="#e74c3c",
        background="#f5f5f5"
    )
    style.configure(
        "Status.TLabel",
        font=("微软雅黑", 11),
        foreground="#27ae60",
        background="#f5f5f5"
    )
    style.configure(
        "Section.TFrame",
        background="#ffffff",
        borderwidth=3,
        relief="groove",
        padding=5
    )
    style.configure(
        "Photo.TFrame",
        background="#ffffff",
        borderwidth=2,
        relief="solid",
        foreground="#333333"
    )
    return style


# ===================== GUI主窗口 =====================
# ===================== GUI主窗口（完整替换原有DeerDetectionGUI类） =====================
class DeerDetectionGUI:
    def __init__(self, root, user_role):
        self.root = root
        self.root.title("鹿群动态识别系统（规范版）")
        self.root.geometry("1300x800")
        self.root.resizable(True, True)

        # 新增：接收登录角色，用于权限控制
        self.user_role = user_role
        # 基础属性（保留原有）
        self.current_photo_name = ""
        self.current_original_path = ""
        self.current_marked_path = ""
        self.cap = None
        self.is_detecting = False
        self.prev_frame = None
        self.video_fps = 25
        self.frame_delay = int(1000 / self.video_fps)
        # 新增：历史记录数据（时间、类型、路径、计数）
        self.history_data = []
        # 新增：视频最终帧和计数
        self.final_video_frame = None
        self.final_deer_count = 0

        self.style = init_style()

        # ===================== 1. 顶部按钮栏（含权限控制+新增退出按钮） =====================
        self.btn_frame = ttk.Frame(root, style="Section.TFrame")
        self.btn_frame.pack(pady=20, fill=tk.X, padx=30)

        # 摄像头检测按钮（所有角色可见）
        self.btn_camera = ttk.Button(self.btn_frame, text="启动摄像头检测", command=self.start_camera,
                                     style="Custom.TButton")
        self.btn_camera.pack(side=tk.LEFT, padx=10, pady=10)

        # 视频检测按钮（所有角色可见）
        self.btn_video = ttk.Button(self.btn_frame, text="打开演示视频", command=self.open_video,
                                    style="Custom.TButton")
        self.btn_video.pack(side=tk.LEFT, padx=10, pady=10)

        # 停止检测按钮（所有角色可见）
        self.btn_stop = ttk.Button(self.btn_frame, text="停止检测", command=self.stop_detect, style="Custom.TButton")
        self.btn_stop.pack(side=tk.LEFT, padx=10, pady=10)

        # 上传照片按钮（仅管理员可见）
        self.btn_upload = ttk.Button(self.btn_frame, text="上传鹿群照片", command=self.upload_photo,
                                     style="Custom.TButton")
        if self.user_role == "admin":
            self.btn_upload.pack(side=tk.LEFT, padx=10, pady=10)

        # 处理照片按钮（仅管理员可见）
        self.btn_process = ttk.Button(self.btn_frame, text="处理照片", command=self.process_photo,
                                      style="Custom.TButton")
        if self.user_role == "admin":
            self.btn_process.pack(side=tk.LEFT, padx=10, pady=10)

        # 导出报告按钮（仅管理员可见）
        self.btn_report = ttk.Button(self.btn_frame, text="导出报告", command=self.export_report,
                                     style="Custom.TButton")
        if self.user_role == "admin":
            self.btn_report.pack(side=tk.LEFT, padx=10, pady=10)

        # ===================== 新增：退出系统按钮（所有角色可见，最右侧） =====================
        self.btn_exit = ttk.Button(self.btn_frame, text="退出系统", command=self.exit_system,
                                   style="Custom.TButton")
        self.btn_exit.pack(side=tk.RIGHT, padx=20, pady=10)

        # ===================== 2. 实时状态卡片（新增） =====================
        self.status_card_frame = ttk.Frame(root, style="Section.TFrame")
        self.status_card_frame.pack(pady=10, fill=tk.X, padx=30)

        # 卡片1：当前检测类型
        self.card1 = ttk.Frame(self.status_card_frame, style="Section.TFrame")
        self.card1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        ttk.Label(self.card1, text="当前检测类型", style="Title.TLabel").pack(pady=5)
        self.label_current_type = ttk.Label(self.card1, text="未检测", style="Status.TLabel")
        self.label_current_type.pack(pady=5)

        # 卡片2：检测帧率
        self.card2 = ttk.Frame(self.status_card_frame, style="Section.TFrame")
        self.card2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        ttk.Label(self.card2, text="检测帧率", style="Title.TLabel").pack(pady=5)
        self.label_fps = ttk.Label(self.card2, text="0 FPS", style="Status.TLabel")
        self.label_fps.pack(pady=5)

        # ===================== 3. 核心标签页（含子标签页+帮助+历史记录） =====================
        self.tab_control = ttk.Notebook(root, style="Custom.TNotebook")
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

        # 标签页1：实时检测（嵌套子标签页：视频/摄像头） =====================
        self.tab_detect = ttk.Frame(self.tab_control, style="Section.TFrame")
        self.tab_control.add(self.tab_detect, text="实时检测")

        # 子标签页控制器
        self.sub_tab_detect = ttk.Notebook(self.tab_detect, style="Custom.TNotebook")
        self.sub_tab_detect.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 子标签1：视频检测
        self.sub_tab_video = ttk.Frame(self.sub_tab_detect, style="Section.TFrame")
        self.sub_tab_detect.add(self.sub_tab_video, text="视频检测")
        # 视频检测画面
        ttk.Label(self.sub_tab_video, text="视频检测（单次播放+保留最后一帧）", style="Title.TLabel").pack(pady=10)
        self.detect_frame = ttk.Frame(self.sub_tab_video, style="Section.TFrame")
        self.detect_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        self.detect_frame.pack_propagate(False)
        self.label_detect = ttk.Label(
            self.detect_frame,
            text="请点击顶部“打开演示视频”",
            font=("微软雅黑", 12),
            anchor="center",
            background="#ffffff"
        )
        self.label_detect.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        # 视频检测状态栏
        self.detect_status_frame = ttk.Frame(self.sub_tab_video, style="Section.TFrame")
        self.detect_status_frame.pack(fill=tk.X, padx=30, pady=15)
        self.label_detect_count = ttk.Label(self.detect_status_frame, text="当前移动鹿数量：0 个", style="Count.TLabel")
        self.label_detect_count.pack(side=tk.LEFT, padx=30, pady=10)
        self.label_detect_status = ttk.Label(self.detect_status_frame, text="状态：未检测", style="Status.TLabel")
        self.label_detect_status.pack(side=tk.LEFT, padx=30, pady=10)

        # 子标签2：摄像头检测
        self.sub_tab_camera = ttk.Frame(self.sub_tab_detect, style="Section.TFrame")
        self.sub_tab_detect.add(self.sub_tab_camera, text="摄像头检测")
        # 摄像头检测画面
        ttk.Label(self.sub_tab_camera, text="摄像头实时检测（移动目标框选）", style="Title.TLabel").pack(pady=10)
        self.camera_frame = ttk.Frame(self.sub_tab_camera, style="Section.TFrame")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        self.camera_frame.pack_propagate(False)
        self.label_camera = ttk.Label(
            self.camera_frame,
            text="请点击顶部“启动摄像头检测”",
            font=("微软雅黑", 12),
            anchor="center",
            background="#ffffff"
        )
        self.label_camera.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        # 摄像头检测状态栏（复用视频检测的计数/状态标签，避免重复）
        self.camera_status_frame = ttk.Frame(self.sub_tab_camera, style="Section.TFrame")
        self.camera_status_frame.pack(fill=tk.X, padx=30, pady=15)
        self.label_camera_count = ttk.Label(self.camera_status_frame, text="当前移动鹿数量：0 个", style="Count.TLabel")
        self.label_camera_count.pack(side=tk.LEFT, padx=30, pady=10)
        self.label_camera_status = ttk.Label(self.camera_status_frame, text="状态：未检测", style="Status.TLabel")
        self.label_camera_status.pack(side=tk.LEFT, padx=30, pady=10)

        # 标签页2：鹿群照片处理（优化：“选择照片”更扁+列表更大） =====================
        self.tab_photo = ttk.Frame(self.tab_control, style="Section.TFrame")
        self.tab_control.add(self.tab_photo, text="鹿群照片处理")

        self.photo_display_frame = ttk.Frame(self.tab_photo, style="Section.TFrame")
        self.photo_display_frame.pack(fill=tk.BOTH, expand=True, pady=15)

        self.left_photo_container = ttk.Frame(self.photo_display_frame, style="Section.TFrame")
        self.left_photo_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)
        self.left_photo_container.pack_propagate(False)
        self.left_photo_container.configure(width=BASE_PHOTO_WIDTH, height=BASE_PHOTO_HEIGHT + 80)

        ttk.Label(self.left_photo_container, text="原始鹿群照片", style="Title.TLabel").pack(pady=10)
        self.original_photo_frame = ttk.Frame(self.left_photo_container, style="Photo.TFrame")
        self.original_photo_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.original_photo_frame.configure(width=BASE_PHOTO_WIDTH, height=BASE_PHOTO_HEIGHT)
        self.original_photo_frame.pack_propagate(False)

        self.label_original = ttk.Label(
            self.original_photo_frame,
            text="",
            anchor="center",
            background="#ffffff"
        )
        self.label_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.right_photo_container = ttk.Frame(self.photo_display_frame, style="Section.TFrame")
        self.right_photo_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)
        self.right_photo_container.pack_propagate(False)
        self.right_photo_container.configure(width=BASE_PHOTO_WIDTH, height=BASE_PHOTO_HEIGHT + 80)

        self.label_marked_title = ttk.Label(
            self.right_photo_container,
            text="处理后照片（鹿数量：0只）",
            style="Title.TLabel"
        )
        self.label_marked_title.pack(pady=10)
        self.marked_photo_frame = ttk.Frame(self.right_photo_container, style="Photo.TFrame")
        self.marked_photo_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.marked_photo_frame.configure(width=BASE_PHOTO_WIDTH, height=BASE_PHOTO_HEIGHT)
        self.marked_photo_frame.pack_propagate(False)

        self.label_marked = ttk.Label(
            self.marked_photo_frame,
            text="",
            anchor="center",
            background="#ffffff"
        )
        self.label_marked.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 优化：“选择照片”更扁+列表更大
        self.photo_list_container = ttk.Frame(self.tab_photo, style="Section.TFrame")
        self.photo_list_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # “选择照片”标签：pady=5更扁
        ttk.Label(self.photo_list_container, text="选择照片", style="Title.TLabel").pack(pady=5)

        # 照片列表框：height=8更大，填充容器宽高
        self.photo_listbox = tk.Listbox(
            self.photo_list_container,
            height=8,
            font=("微软雅黑", 10),
            relief="solid",
            borderwidth=1,
            bg="white",
            selectbackground="#3498db"
        )
        self.photo_listbox.pack(fill=tk.BOTH, expand=True, pady=5, padx=20)
        self.photo_listbox.bind('<<ListboxSelect>>', self.select_photo_from_list)

        # 标签页3：历史记录（新增） =====================
        self.tab_history = ttk.Frame(self.tab_control, style="Section.TFrame")
        self.tab_control.add(self.tab_history, text="历史记录")

        ttk.Label(self.tab_history, text="检测历史记录（点击加载）", style="Title.TLabel").pack(pady=10)
        # 历史记录列表框
        self.history_listbox = tk.Listbox(self.tab_history, font=("微软雅黑", 10), height=12, bg="#fff")
        self.history_listbox.pack(fill=tk.X, padx=20, pady=5)
        # 加载按钮
        ttk.Button(self.tab_history, text="加载选中记录", command=self.load_history, style="Custom.TButton").pack(
            pady=10)

        # 标签页4：帮助中心（新增） =====================
        self.tab_help = ttk.Frame(self.tab_control, style="Section.TFrame")
        self.tab_control.add(self.tab_help, text="帮助中心")

        # 帮助内容（不可编辑的Text组件）
        help_content = """
# 鹿群动态识别系统 - 操作指引
1. 登录说明：
   - 管理员账号：admin / 密码：123456（全功能权限）
   - 普通用户账号：user / 密码：123456（仅查看权限）

2. 核心功能操作：
   ▶ 视频检测：
      1. 点击顶部“打开演示视频”，选择MP4/AVI格式文件；
      2. 视频单次自动播放，移动鹿会被红色框选；
      3. 播放结束后保留最后一帧，显示最终计数。
   ▶ 摄像头检测：
      1. 点击顶部“启动摄像头检测”；
      2. 对准移动目标（如鹿群图片），实时框选计数；
      3. 点击“停止检测”可结束。
   ▶ 照片处理（管理员）：
      1. 点击“上传鹿群照片”，选择JPG/PNG文件；
      2. 在“鹿群照片处理”标签页选择照片；
      3. 点击“处理照片”查看结果。
   ▶ 导出报告（管理员）：
      1. 检测完成后点击“导出报告”；
      2. 选择保存路径，生成TXT格式报告。
   ▶ 退出系统：
      1. 点击顶部右侧“退出系统”；
      2. 确认后自动释放资源并关闭程序。

3. 常见问题：
   Q1：视频无法播放？
   A1：确保视频格式为MP4/AVI，关闭其他占用视频的程序。
   Q2：计数不准？
   A2：管理员可调整参数：CONTOUR_AREA_THRESHOLD（越小检测越灵敏）。
   Q3：摄像头打不开？
   A3：关闭微信/QQ等占用摄像头的程序，检查硬件连接。

4. 参数说明：
   - 帧差阈值（DIFF_THRESHOLD）：20-50，值越小越灵敏；
   - 轮廓面积阈值（CONTOUR_AREA_THRESHOLD）：100-500，过滤小噪声；
   - 帧率（FPS）：默认25，匹配多数视频播放速度。
"""
        self.help_text = tk.Text(self.tab_help, font=("微软雅黑", 10), wrap=tk.WORD, bg="#fff")
        self.help_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.help_text.insert(tk.END, help_content)
        self.help_text.config(state=tk.DISABLED)  # 禁止编辑

        # 标签页5：系统日志（保留原有） =====================
        self.tab_log = ttk.Frame(self.tab_control, style="Section.TFrame")
        self.tab_control.add(self.tab_log, text="系统日志")

        ttk.Label(self.tab_log, text="操作日志", style="Title.TLabel").pack(pady=15)
        self.log_text = tk.Text(
            self.tab_log,
            font=("微软雅黑", 10),
            relief="solid",
            borderwidth=1,
            bg="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # 初始化加载照片
        self.load_original_photos()
        self.add_log(f"系统启动成功，当前角色：{self.user_role}")
        self.add_log(f"照片路径：{ORIGINAL_DIR}")

    # ===================== 新增：退出系统功能（完整实现） =====================
    def exit_system(self):
        """退出系统：确认提示+释放资源+关闭窗口"""
        # 弹出确认框，防止误操作
        confirm = messagebox.askyesno("确认退出", "是否确定退出鹿群识别系统？\n退出后将释放所有检测资源。")
        if confirm:
            # 第一步：释放摄像头/视频资源（复用停止检测逻辑）
            self.stop_detect()
            # 第二步：记录退出日志
            self.add_log("用户主动退出系统，程序正常关闭")
            # 第三步：关闭主窗口，结束程序
            self.root.destroy()

    # ===================== 以下为原有基础功能（无需修改，直接保留） =====================
    def load_original_photos(self):
        self.photo_listbox.delete(0, tk.END)
        for file_name in os.listdir(ORIGINAL_DIR):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                self.photo_listbox.insert(tk.END, file_name)

    def upload_photo(self):
        file_paths = filedialog.askopenfilenames(
            title="选择原始鹿群照片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if not file_paths:
            return

        success_num = 0
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            target_path = os.path.join(ORIGINAL_DIR, file_name)

            if os.path.exists(target_path):
                self.add_log(f"跳过已存在文件：{file_name}")
                continue

            try:
                with open(file_path, "rb") as f_src, open(target_path, "wb") as f_dst:
                    f_dst.write(f_src.read())
                self.photo_listbox.insert(tk.END, file_name)
                success_num += 1
            except Exception as e:
                self.add_log(f"上传失败：{file_name} - {str(e)}")
                messagebox.showerror("错误", f"上传{file_name}失败：{str(e)}")

        self.add_log(f"成功上传{success_num}张照片")
        messagebox.showinfo("提示", f"成功上传{success_num}张照片！")

    def select_photo_from_list(self, event):
        selected = self.photo_listbox.curselection()
        if not selected:
            return

        file_name = self.photo_listbox.get(selected[0])
        self.current_photo_name = os.path.splitext(file_name)[0]
        self.current_original_path = os.path.join(ORIGINAL_DIR, file_name)
        self.current_marked_path = os.path.join(MARKED_DIR, f"{self.current_photo_name}_marked.jpg")

        self.show_original_photo()
        self.add_log(f"选择照片：{file_name}")

    def show_original_photo(self):
        self.label_original.config(text="", image="")
        if not os.path.exists(self.current_original_path):
            self.label_original.config(text="未找到原始照片")
            return

        try:
            img = Image.open(self.current_original_path)
            img.thumbnail((BASE_PHOTO_WIDTH, BASE_PHOTO_HEIGHT), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (BASE_PHOTO_WIDTH, BASE_PHOTO_HEIGHT), color="#ffffff")
            x = (BASE_PHOTO_WIDTH - img.width) // 2
            y = (BASE_PHOTO_HEIGHT - img.height) // 2
            canvas.paste(img, (x, y))

            tk_img = ImageTk.PhotoImage(canvas)
            self.label_original.config(image=tk_img, text="")
            self.label_original.image = tk_img
        except Exception as e:
            self.label_original.config(text="加载失败")
            self.add_log(f"加载原始图失败：{str(e)}")
            messagebox.showerror("错误", f"加载原始图失败：{str(e)}")

    def process_photo(self):
        if not self.current_photo_name:
            messagebox.showwarning("警告", "请先选择原始照片！")
            return

        self.label_marked.config(text="", image="")
        if not os.path.exists(self.current_marked_path):
            self.label_marked.config(text="未找到对应标记图")
            self.add_log(f"未找到标记图：{self.current_marked_path}")
            messagebox.showerror("错误", f"未找到标记图：{self.current_photo_name}_marked.jpg")
            return

        try:
            img = Image.open(self.current_marked_path)
            img.thumbnail((BASE_PHOTO_WIDTH, BASE_PHOTO_HEIGHT), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (BASE_PHOTO_WIDTH, BASE_PHOTO_HEIGHT), color="#ffffff")
            x = (BASE_PHOTO_WIDTH - img.width) // 2
            y = (BASE_PHOTO_HEIGHT - img.height) // 2
            canvas.paste(img, (x, y))

            tk_img = ImageTk.PhotoImage(canvas)
            self.label_marked.config(image=tk_img, text="")
            self.label_marked.image = tk_img

            deer_count = PHOTO_COUNT_DICT.get(self.current_photo_name, 0)
            self.label_marked_title.config(text=f"处理后照片（鹿数量：{deer_count}只）")

            self.add_log(f"处理照片完成：{self.current_photo_name}，鹿数量：{deer_count}")
            messagebox.showinfo("提示", "照片处理完成！")
        except Exception as e:
            self.label_marked.config(text="加载失败")
            self.add_log(f"加载标记图失败：{str(e)}")
            messagebox.showerror("错误", f"加载标记图失败：{str(e)}")

    # ===================== 以下为摄像头检测（新增状态更新+历史记录） =====================
    def start_camera(self):
        self.stop_detect()
        self.prev_frame = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.add_log("无法打开摄像头（请关闭占用程序）")
            messagebox.showerror("错误", "无法打开摄像头！")
            return

        self.is_detecting = True
        self.label_camera_status.config(text="状态：检测中（摄像头）")
        self.label_current_type.config(text="摄像头检测")
        self.add_log("启动摄像头检测")
        self.update_camera_frame()

    def update_camera_frame(self):
        if not self.is_detecting or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detect()
            return

        original_frame = frame.copy()
        frame_width = self.camera_frame.winfo_width()
        frame_height = self.camera_frame.winfo_height()
        if frame_width < 100 or frame_height < 100:
            frame_width = 640
            frame_height = 480

        # 帧差法检测
        target_count = 0
        if self.prev_frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, gray_prev)
            _, diff_thresh = cv2.threshold(frame_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD:
                    target_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

        # 更新计数和帧率
        self.label_camera_count.config(text=f"当前移动鹿数量：{target_count} 个")
        self.label_fps.config(text=f"{self.video_fps} FPS")

        # 保存当前帧
        self.prev_frame = frame.copy()

        # 更新界面画面
        frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.label_camera.config(image=tk_img, text="")
        self.label_camera.image = tk_img

        self.root.after(self.frame_delay, self.update_camera_frame)

    # ===================== 以下为视频检测（新增历史记录保存） =====================
    def open_video(self):
        self.stop_detect()
        self.prev_frame = None
        self.final_video_frame = None
        self.final_deer_count = 0

        video_path = filedialog.askopenfilename(
            title="选择演示视频（含移动鹿）",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.add_log(f"无法打开视频：{video_path}")
            messagebox.showerror("错误", "无法打开视频！")
            return

        # 获取视频实际帧率
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 25
        self.frame_delay = int(1000 / self.video_fps)

        self.is_detecting = True
        self.label_detect_status.config(text="状态：检测中（视频）")
        self.label_current_type.config(text="视频检测")
        self.add_log(f"打开演示视频：{os.path.basename(video_path)}（帧率：{self.video_fps}）")
        self.update_video_frame(video_path)  # 传入视频路径，用于保存历史

    def update_video_frame(self, video_path):
        if not self.is_detecting or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # 视频播放完成：保存历史+显示最后一帧
            self.is_detecting = False
            self.cap.release()
            self.cap = None
            # 保存历史记录
            self.save_history("视频检测", video_path, self.final_deer_count)
            # 显示最后一帧
            if self.final_video_frame is not None:
                frame_width = self.detect_frame.winfo_width()
                frame_height = self.detect_frame.winfo_height()
                if frame_width < 100 or frame_height < 100:
                    frame_width = 640
                    frame_height = 480

                final_rgb = cv2.cvtColor(self.final_video_frame, cv2.COLOR_BGR2RGB)
                final_img = Image.fromarray(final_rgb)
                final_img.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
                final_tk_img = ImageTk.PhotoImage(final_img)

                self.label_detect.config(image=final_tk_img, text="")
                self.label_detect.image = final_tk_img
                self.label_detect_count.config(text=f"最终移动鹿数量：{self.final_deer_count} 个")
                self.label_detect_status.config(text="状态：视频已结束")

                self.add_log(f"视频播放完成，最终计数：{self.final_deer_count}")
                messagebox.showinfo("视频结束", f"最终检测到移动鹿数量：{self.final_deer_count} 个")
            return

        original_frame = frame.copy()
        frame_width = self.detect_frame.winfo_width()
        frame_height = self.detect_frame.winfo_height()
        if frame_width < 100 or frame_height < 100:
            frame_width = 640
            frame_height = 480

        # 帧差法检测
        target_count = 0
        if self.prev_frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, gray_prev)
            _, diff_thresh = cv2.threshold(frame_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD:
                    target_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

        # 更新计数和帧率
        self.label_detect_count.config(text=f"当前移动鹿数量：{target_count} 个")
        self.label_fps.config(text=f"{int(self.video_fps)} FPS")

        # 保存最后一帧和计数
        self.final_video_frame = original_frame.copy()
        self.final_deer_count = target_count

        # 保存当前帧
        self.prev_frame = frame.copy()

        # 更新界面画面
        frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.label_detect.config(image=tk_img, text="")
        self.label_detect.image = tk_img

        self.root.after(self.frame_delay, self.update_video_frame, video_path)

    # ===================== 以下为新增：历史记录相关函数 =====================
    def save_history(self, detect_type, file_path, count):
        """保存检测历史到列表和界面"""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history_data.append((time_str, detect_type, file_path, count))
        # 显示到列表框
        self.history_listbox.insert(tk.END, f"{time_str} | {detect_type} | 计数：{count}")

    def load_history(self):
        """加载选中的历史记录"""
        selected = self.history_listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一条历史记录！")
            return

        idx = selected[0]
        time_str, detect_type, file_path, count = self.history_data[idx]
        # 视频记录：重新打开视频
        if detect_type == "视频检测" and os.path.exists(file_path):
            self.open_video_from_history(file_path)
        # 显示提示
        messagebox.showinfo("历史记录详情",
                            f"时间：{time_str}\n类型：{detect_type}\n文件：{os.path.basename(file_path)}\n计数：{count}")

    def open_video_from_history(self, file_path):
        """从历史记录重新打开视频"""
        self.stop_detect()
        self.prev_frame = None
        self.final_video_frame = None
        self.final_deer_count = 0

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.add_log(f"无法加载历史视频：{file_path}")
            messagebox.showerror("错误", "无法加载历史视频！")
            return

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 25
        self.frame_delay = int(1000 / self.video_fps)

        self.is_detecting = True
        self.label_detect_status.config(text="状态：检测中（历史视频）")
        self.label_current_type.config(text="历史视频检测")
        self.add_log(f"加载历史视频：{os.path.basename(file_path)}")
        self.update_video_frame(file_path)

    # ===================== 以下为原有停止检测/导出报告/日志函数（无需修改） =====================
    def stop_detect(self):
        self.is_detecting = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.prev_frame = None
        self.label_current_type.config(text="未检测")
        self.label_fps.config(text="0 FPS")

        # 重置摄像头界面
        self.label_camera.config(text="请点击顶部“启动摄像头检测”", image="")
        self.label_camera_count.config(text="当前移动鹿数量：0 个")
        self.label_camera_status.config(text="状态：未检测")

        # 重置视频界面（保留最后一帧）
        if self.final_video_frame is None:
            self.label_detect.config(text="请点击顶部“打开演示视频”", image="")
            self.label_detect_count.config(text="当前移动鹿数量：0 个")
        self.label_detect_status.config(text="状态：未检测")

        self.add_log("停止所有检测，资源已释放")

    def export_report(self):
        if not self.current_photo_name and self.final_deer_count == 0:
            messagebox.showwarning("警告", "暂无检测数据！")
            return

        save_path = filedialog.asksaveasfilename(
            title="导出检测报告",
            defaultextension=".txt",
            filetypes=[("文本文件 (*.txt)", "*.txt"), ("所有文件", "*.*")],
            initialfile=f"鹿群检测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        if not save_path:
            self.add_log("用户取消导出报告")
            return

        try:
            report_content = f"""
===================== 鹿群动态识别系统检测报告 =====================
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
当前用户角色：{self.user_role}
==================================================================
"""
            if self.current_photo_name:
                deer_count = PHOTO_COUNT_DICT.get(self.current_photo_name, 0)
                report_content += f"""
【照片检测结果】
照片名称：{self.current_photo_name}
鹿群数量：{deer_count}只
原始路径：{self.current_original_path}
标记路径：{self.current_marked_path}
"""
            if self.final_deer_count > 0:
                report_content += f"""
【视频检测结果】
最终移动鹿数量：{self.final_deer_count}只
检测参数：
  - 帧差阈值：{DIFF_THRESHOLD}
  - 轮廓面积阈值：{CONTOUR_AREA_THRESHOLD}
  - 检测帧率：{self.video_fps} FPS
"""
            report_content += "\n\n===================== 报告结束 ====================="

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.add_log(f"报告导出成功：{save_path}")
            messagebox.showinfo("成功", f"报告已导出至：{save_path}")
        except Exception as e:
            self.add_log(f"导出报告失败：{str(e)}")
            messagebox.showerror("错误", f"导出报告失败：{str(e)}")

    def add_log(self, message):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{time_str}] {message}\n")
        self.log_text.see(tk.END)  # 滚动到最后


# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 先显示登录窗口
    login_root = tk.Tk()
    login_window = LoginWindow(login_root)
    login_root.mainloop()

    # 登录成功后显示主窗口
    if login_window.user_role:
        root = tk.Tk()
        app = DeerDetectionGUI(root, login_window.user_role)
        root.mainloop()