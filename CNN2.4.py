# -*- coding:utf-8 -*-

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 构建网络 150*150*3 --> 148*148*32 --> 74*74*32 --> 72*72*64 --> 36*36*64 --> 34*34*128 -->
#          17*17*128 --> 15*15*128 --> 7*7*128 --> 6272 --> 512 --> 1
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# print(model.summary())

# 编译模型 使用RMSprop优化器，因为网络最后一层是单一sigmoid 单元，使用二院交叉熵
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 数据预处理
"""
1.读取图像文件
2.将JPEG文件解码为RGB像素网格
3.将像素网格转换为浮点数张量
4.将像素值（0-255）缩放到（0,1]区间
"""
train_dir = r'D:\datasets\cats_and_dogs_small\train'
validation_dir = r'D:\datasets\cats_and_dogs_small\validation'

# 对训练集做数据增强
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

# 使用生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,                # 表示从生成器中取出多少批次后进行下一轮，2000/20=100
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

# 保存模型
model.save('cats_and_dogs_small_2.h5')

# 绘制训练过程中的损失曲线和精度曲线
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


