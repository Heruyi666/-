import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearR_myselff import LinearRegression
from data.prepare_for_training import Data_prepare 
from matplotlib import pylab

(data_processed,num_features) = Data_prepare()
x_train,x_test,y_train,y_test = data_processed

# plt.scatter(x_train,y_train,label = 'Train data')
# plt.scatter(x_test,y_test,label = 'test data')
# plt.legend()
# plt.show()
num_epoch = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train,y_train)
theta,cost_history = linear_regression.train(learning_rate,num_epoch)
print('theta的值是',theta)
print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])
plt.plot(range(num_epoch),cost_history)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions,predictions_num)
plt.scatter(x_train[:,1],y_train,label = 'Train data')
plt.scatter(x_test[:,1],y_test,label = 'test data')
plt.plot(x_predictions,y_predictions,'r',label = 'prediction')
plt.legend()
plt.show()

