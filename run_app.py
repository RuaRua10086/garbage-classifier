import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# 定义模型路径和类别名称
MODEL_PATH = 'garbage_classifier_model.keras'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_HEIGHT = 224
IMG_WIDTH = 224

class GarbageClassifierApp:
    """垃圾分类器GUI应用程序主类"""
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("智能垃圾分类识别器")

        # 设置窗口大小和位置
        window_width = 700
        window_height = 650
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(False, False) # 禁止调整窗口大小

        # 加载模型
        self.model = self.load_model()
        if self.model is None:
            self.root.destroy() # 如果模型加载失败，则关闭应用
            return

        # 创建界面组件
        self.create_widgets()

    def load_model(self):
        """加载训练好的Keras模型"""
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("错误", f"模型文件 '{MODEL_PATH}' 未找到。\n请先运行 'train_model.py' 来生成模型。")
            return None
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("模型加载成功！")
            return model
        except Exception as e:
            messagebox.showerror("模型加载错误", f"无法加载模型文件: {e}")
            return None

    def create_widgets(self):
        """创建GUI界面上的所有组件"""
        # 标题标签
        title_label = tk.Label(self.root, text="智能垃圾分类识别器", font=("Arial", 20, "bold"), pady=15)
        title_label.pack()

        # 用于显示图片的框架
        image_frame = tk.Frame(self.root, relief=tk.SUNKEN, borderwidth=2, width=450, height=450)
        image_frame.pack(pady=10)
        image_frame.pack_propagate(False) # 防止框架因内部组件而改变大小
        self.image_label = tk.Label(image_frame, text="请上传图片", font=("Arial", 14))
        self.image_label.pack(expand=True)

        # 结果显示标签
        self.result_label = tk.Label(self.root, text="预测结果", font=("Arial", 16, "bold"), fg="#000080", pady=10)
        self.result_label.pack()

        # 上传按钮
        upload_button = tk.Button(self.root, text="上传并识别图片", font=("Arial", 14), command=self.classify_image, bg="#4CAF50", fg="white", relief=tk.RAISED)
        upload_button.pack(pady=20, ipadx=10, ipady=5)

    def classify_image(self):
        """打开文件对话框，加载、显示并预测图片"""
        # 1. 打开文件对话框让用户选择图片
        filepath = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return  # 如果用户取消选择，则不执行任何操作

        # 2. 在GUI上显示用户选择的图片
        self.display_image(filepath)

        # 3. 对图片进行预测
        try:
            # 加载并预处理图片以符合模型输入要求
            img = tf.keras.utils.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # 创建一个批次

            # 使用模型进行预测
            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            # 4. 解析预测结果并更新标签
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            result_text = f"预测类别: {predicted_class.capitalize()}\n置信度: {confidence:.2f}%"
            self.result_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("识别错误", f"图片识别过程中发生错误: {e}")
            self.result_label.config(text="图片识别失败")
            
    def display_image(self, filepath):
        """在GUI的image_label上显示图片"""
        try:
            # 使用Pillow打开图片
            img = Image.open(filepath)
            # 调整图片大小以适应GUI框架，同时保持纵横比
            img.thumbnail((450, 450))
            # 转换为Tkinter兼容的格式
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="") # 清空"请上传图片"的文本
            self.image_label.image = photo  # 必须保持对该对象的引用，否则图片不会显示
        except Exception as e:
            messagebox.showerror("图片显示错误", f"无法显示图片: {e}")
            self.image_label.config(image=None, text="无法显示图片")


if __name__ == "__main__":
    # 创建主窗口并运行应用
    main_window = tk.Tk()
    app = GarbageClassifierApp(main_window)
    main_window.mainloop()