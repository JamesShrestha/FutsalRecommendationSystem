import numpy as np
import pandas as pd
import time
from random import normalvariate

class SGD:
    def __init__(self, learning_rate=0.01, regularization_rate=0.01, num_features=7, max_epoch = 500):
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.num_features = num_features
        self.max_epoch = max_epoch
    
    def train_new_user(self, newData):
        new_user_bias = np.mean(newData[newData.nonzero()]) - self.overall_mean
        nuser_feature_matrix = np.array([normalvariate(0,1) for _ in range(7)])
        futsalIds = newData.nonzero()[0]
        for epoch in range(self.max_epoch):
          for fid in futsalIds:
            predicted = self.overall_mean + new_user_bias + self.bias_futsal[fid] + np.dot(self.Q[fid].T, nuser_feature_matrix)
            error = newData[fid] - predicted
            new_user_bias = new_user_bias + self.learning_rate * (error - self.regularization_rate * new_user_bias)
            nuser_feature_matrix = nuser_feature_matrix + self.learning_rate * (error * self.Q[fid] - self.regularization_rate * nuser_feature_matrix)
        f_bias_matrix = np.array(self.bias_futsal).T
        u_bias_matrix = new_user_bias * np.ones(len(self.bias_futsal))
        g_mean_matrix = self.overall_mean * np.ones(len(self.bias_futsal))
        d_matrix = np.dot(self.Q, nuser_feature_matrix.T)
        prediction_matrix = f_bias_matrix + u_bias_matrix + g_mean_matrix + d_matrix
        return prediction_matrix

    def initialize(self, trainData):
        self.num_users = trainData.shape[1]
        self.num_futsals = trainData.shape[0]
        self.bias_user = [normalvariate(0,1) for _ in range(self.num_users)]
        self.bias_futsal = [normalvariate(0,1) for _ in range(self.num_futsals)]
        self.initial_user_bias = self.bias_user
        self.initial_futsal_bias = self.bias_futsal
        self.overall_mean = np.mean(trainData.values[trainData.values.nonzero()])
        self.fids, self.uids = trainData.values.nonzero()
        self.errorTrain = []
        self.errorTest = []
        return self
    
    def get_initial_biases(self):
        return self.initial_user_bias, self.initial_futsal_bias
      
    def prediction(self, uid, fid):
        return self.overall_mean + self.bias_user[uid] + self.bias_futsal[fid] + np.dot(self.Q[fid].T, self.P[uid])
    
    def actual_prediction_matrix(self,trainData):
        futsal_bias_matrix = np.array([self.bias_futsal for _ in range(len(self.bias_user))]).T
        user_bias_matrix = np.array([self.bias_user for _ in range(len(self.bias_futsal))])
        global_mean_matrix = self.overall_mean * np.ones(trainData.shape)
        dot_matrix = np.dot(self.Q, self.P.T)
        return futsal_bias_matrix + user_bias_matrix + global_mean_matrix + dot_matrix
        
    def mean_squared_error(self, prediction, truth):
        num_cond = len(prediction)
        sq_error = 0
        for i in range(num_cond):
            sq_error += (prediction[i] - truth[i]) ** 2
        return sq_error / num_cond

    def rmse(self, prediction, actual):
        prediction = prediction[actual.values.nonzero()].flatten()
        actual = actual.values[actual.values.nonzero()].flatten()
        return np.sqrt(self.mean_squared_error(prediction, actual))
    
    def fit(self, trainData, testData, user_feature_matrix, futsal_feature_matrix):
        self.initialize(trainData)
        self.P = user_feature_matrix
        self.Q = futsal_feature_matrix
        start_time = time.time()
        for epoch in range(self.max_epoch):
            for uid, fid in zip(self.uids, self.fids):
                error = trainData.values[fid][uid] - self.prediction(uid, fid)
                self.bias_user[uid] = self.bias_user[uid] + self.learning_rate * (error - self.regularization_rate * self.bias_user[uid])
                self.bias_futsal[fid] = self.bias_futsal[fid] + self.learning_rate * (error - self.regularization_rate * self.bias_futsal[fid])
                self.P[uid] = self.P[uid] + self.learning_rate * (error * self.Q[fid] - self.regularization_rate * self.P[uid])
                self.Q[fid] = self.Q[fid] + self.learning_rate * (error * self.P[uid] - self.regularization_rate * self.Q[fid])
            current_prediction = self.actual_prediction_matrix(trainData)
            self.errorTrain.append(self.rmse(current_prediction, trainData))
            self.errorTest.append(self.rmse(current_prediction, testData))
        elapsed_time = time.time() - start_time
        print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        return self.P, self.Q, self.bias_user, self.bias_futsal, self.overall_mean
    
    def print_rmse(self):
        print("Training\t\tTesting")
        for i in range(len(self.errorTrain)):
            print(f'{self.errorTrain[i]}\t{self.errorTest[i]}')
    
    def get_rmse(self):
        return self.errorTrain, self.errorTest

    def get_results(self):
        return self.P, self.Q, self.bias_user, self.bias_futsal, self.overall_mean
    
# model = pickle.load(open("recomm-model.pkl","rb"))