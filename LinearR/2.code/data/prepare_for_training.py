
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd 
import numpy as np
import torch


def Data_prepare():
    """
    线性回归:正规方程
    :return:
    """
    # 1.获取数据
    #boston = load_boston()
    data = pd.read_csv('E:\\MLcode\\LinearR\\2.code\\data\\boston_housing_prices.csv')
    # print(boston)
    train_data = data.sample(frac = 0.8)
    test_data = data.drop(train_data.index)
    input_param_name = 'RM'
    output_param_name = 'MEDV'
    x_train = train_data[[input_param_name]].values
    y_train = train_data[[output_param_name]].values
    x_test = test_data[[input_param_name]].values
    y_test = test_data[[output_param_name]].values

    # 2.数据基本处理
    # 2.1 分割数据
    # x_train, x_test, y_train, y_test = train_test_split(boston.RM, boston.MEDV, test_size=0.2)
    # x_train = np.array(x_train).reshape(-1,1)
    # x_test =  np.array(x_test).reshape(-1,1)
    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    num_features = x_train.shape[1]+1
    num_examples = x_train.shape[0]
    num_examples2 = x_test.shape[0]
  
    x_train = np.hstack((np.ones((num_examples,1)),x_train))
    x_test = np.hstack((np.ones((num_examples2,1)),x_test))
    data_processed = x_train,x_test,y_train,y_test
    # data2 = train_data.drop(["MEDV"], axis=1)
    # Train_data = data2.values
    return data_processed,num_features

