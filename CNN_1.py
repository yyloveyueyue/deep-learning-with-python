# -*- coding:utf-8 -*-

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical


(train_image,train_labels),(test_images,test_labels) = mnist.load_data()

# 图像数据预处理
train_image = train_image.reshape((60000, 28, 28, 1))
train_image = train_image.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 模型搭建
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 训练模型
model.fit(train_image, train_labels, epochs=5, batch_size=64)

#评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)