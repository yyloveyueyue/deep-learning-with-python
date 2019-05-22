# -*- coding:utf-8 -*-

"""
s使用VGG16，实现端到端的运行模型
"""
from keras.applications import VGG16
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt


conv_base = VGG16(
    weights='imagenet',          # 指定模型初始化的权重检查点
    include_top=False,            # 是否包含密集连接分类器
    input_shape=(150, 150, 3)      # 输入图像张量的形状
)
# 冻结vgg16 卷积基，训练时保持权重不变
conv_base.trainable = False

# 构建网络
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 生成器数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,                   # 表示随机旋转的角度
    width_shift_range=0.2,               # 图像在水平、垂直方向上平移的范围，是个比例
    height_shift_range=0.2,
    shear_range=0.2,                     # 随机错切变化的角度
    zoom_range=0.2,                      # 图像随机缩放的范围
    horizontal_flip=True,                # 随机将一半图像水平翻转
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = r'D:\datasets\cats_and_dogs_small\train'
validation_dir = r'D:\datasets\cats_and_dogs_small\validation'

train_generator = train_datagen.flow_from_directory(
    # This is the target directory（目标目录）
    train_dir,
    # All images will be resized to 150x150（将所有图像的大小调整为 150×150）
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    # 因为使用了 binary_crossentropy损失，所以需要用二进制标签
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
