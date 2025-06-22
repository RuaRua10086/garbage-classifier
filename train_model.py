import tensorflow as tf
import matplotlib.pyplot as plt
import os

print(f"TensorFlow 版本: {tf.__version__}")

# 数据集路径 (请确保此文件夹与脚本在同一目录下)
DATASET_PATH = 'GarbageClassification'

# 图像参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32

# 训练轮次参数
EPOCHS = 50 

# 3. 加载和预处理数据
print("--- 步骤 1/7: 开始加载数据 ---")

if not os.path.exists(DATASET_PATH):
    print(f"错误: 数据集路径 '{DATASET_PATH}' 不存在。")
    print("请从Kaggle下载数据集并将其解压到与此脚本相同的目录中。")
    exit()

# 使用Keras工具从目录加载数据，并自动划分为80%训练集和20%验证集
try:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"加载数据时出错: {e}")
    print("请确保数据集文件夹结构正确。")
    exit()


# 获取类别名称
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"数据加载完成。共找到 {num_classes} 个类别: {class_names}")

# 优化性能：使用缓存和预取以加速训练过程
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("--- 步骤 2/7: 预处理完成 ---")

print("--- 步骤 3/7: 开始构建模型 ---")

# 数据增强层: 在训练时对图像进行随机变换，提高模型泛化能力
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# 预处理层: 将像素值从[0, 255]缩放到[-1, 1]，以匹配MobileNetV2的输入要求
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 加载MobileNetV2作为基座模型，不包括其顶部的分类层，并使用在ImageNet上预训练的权重
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet'
)

# 冻结基座模型，使其权重不被更新
base_model.trainable = False

# 构建完整模型
inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
# 1. 数据增强
x = data_augmentation(inputs)
# 2. 预处理
x = preprocess_input(x)
# 3. 特征提取 (使用基座模型)
x = base_model(x, training=False)
# 4. 将特征图展平
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# 5. 添加Dropout层以防止过拟合
x = tf.keras.layers.Dropout(0.2)(x)
# 6. 添加分类层
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print("--- 步骤 4/7: 模型构建完成 ---")

# 5. 编译模型
print("--- 步骤 5/7: 编译模型 ---")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 6. 训练模型
print("--- 步骤 6/7: 开始训练模型 ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("模型训练完成。")

# 7. 评估和保存模型
print("--- 步骤 7/7: 评估和保存模型 ---")

# 绘制并保存训练过程中的准确率和损失值曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 保存图表到文件
figure_save_path = 'training_performance_curves.png'
plt.savefig(figure_save_path)
print(f"训练曲线图已保存为: {figure_save_path}")

# 保存训练好的模型
model_save_path = 'garbage_classifier_model.keras'
model.save(model_save_path)
print(f"模型已成功保存为: {model_save_path}")
print("项目完成！")