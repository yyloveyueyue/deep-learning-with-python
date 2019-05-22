# -*- coding:utf-8 -*-

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.0
    return results


(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

# 将训练、测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#将标签向量化
"""
np.array和np.array虽然都可以讲结构数据转化为ndarray，但当原数据为ndarray时，array依旧会copy
一个副本，而asarray不会，所以当原始数据发生变化，asarray也会变化，而array不会
"""
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 构建网络
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# 编译模型
model.compile( optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['acc'])

#将数据分出10000份验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

# 绘制训练损失和验证损失
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()