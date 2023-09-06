import numpy as np
from data.prepare_for_training import Data_prepare





class LinearRegression:
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data = True):
        
        self.data = data
        data_processed,num_features = Data_prepare()
        
        self.theta = np.zeros((num_features,1))
        self.labels = labels
    
    def train(self,alpha,num_epoch):
        """
        训练模块
        """
        
        cost_history = self.gradient_descent(alpha,num_epoch)
        return self.theta,cost_history
    
    def gradient_descent(self,alpha,num_epoch):
        """
        实际迭代模块，迭代num_epoch次
        """
        cost_history = []
        for _ in range(num_epoch):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        """
        梯度下降
        """
        num_examples = self.data.shape[0]
        
        
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta

    def cost_function(self,data,labels):
        """
        计算loss
        """
        
        num_examples = int(data.shape[0])
        delta = LinearRegression.hypothesis(data,self.theta)- labels
        cost = [(1/2)*np.dot(delta.T,delta)]
        cost[0][0] = cost[0][0]/num_examples
        return cost[0][0]


    def hypothesis(data,theta):
        
    
        predictions = np.dot(data,theta)
        return predictions
    
    def get_cost(self,data,labels):
        
        data = data
        return self.cost_function(data,labels)

    def predict(self,data,predictions_num):
        data = np.hstack((np.ones((predictions_num,1)),data))
        predictions = LinearRegression.hypothesis(data,self.theta)
        return predictions