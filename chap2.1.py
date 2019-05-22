# -*- coding:utf-8 -*-

from keras.datasets import mnist
from keras import models

from keras import layers


(train_image,train_labels),(test_images,test_labels) = mnist.load_data()

# print(train_image.shape)
network = models.Sequential()
network.add( layers.Dense(512,activation='relu',input_shape=(28*28,)) )
network.add( layers.Dense( 10, activation = 'softmax' ) )

# 编译步骤
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 图像数据预处理
train_image = train_image.reshape((60000,28*28))
train_image = train_image.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# 对标签分类编码
from keras.utils import  to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(network.fit(train_image,train_labels,epochs=5,batch_size=128))

test_loss, test_acc = network.evaluate(test_images,test_labels)
print('test_acc:', test_acc)