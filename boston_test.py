# -*- coding:utf-8 -*-

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 加载数据，其中训练集(404,13)，测试集(102,13)
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 对数据标准化，注意测试集标准化的均值和标准差都是在训练集上获得的
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

# 构建网络，因为数据集很小，使用很小的网络，避免过拟合
def build_model():
    """
    需要将同一个模型多次实例化，所以用一个函数来构建模型
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))                      # 未使用激活函数，因为这是标量回归

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model


# 使用k折验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
# all_scores = []
all_mae_histores = []

for i in range(k):
    # print('pricessing fold #', i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i+1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i+1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs,batch_size=32,verbose=0)
    # val_mse, val_mae = model.evaluate(val_data,val_targets,verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    # all_scores.append(val_mae)
    all_mae_histores.append(mae_history)

# 计算所有轮次中的k折验证分数平均值
average_mae_history = [ np.mean([x[i] for x in all_mae_histores]) for i in range(num_epochs) ]

# 绘制验证分数
plt.plot(range(1, len(average_mae_history) + 1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()