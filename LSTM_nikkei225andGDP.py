#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:16:02 2017

@author: Norio Yamashita
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.layers.advanced_activations import LeakyReLU

look_back = 10
hidden_neurons = 300

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, 0])      
        dataX.append(xset)
    return np.array(dataX), np.array(dataY)

# invert predictions
def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])


if __name__ == "__main__":

  # データ準備
  dataframe = None
  dataframe = pandas.read_csv('csv/nikkei225_and_GDP.csv',usecols=[1,4])
  dataset = dataframe.values
  dataset = dataset.astype('float32')
  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  
  # split into train and test sets
  train_size = int(len(dataset) * 0.8)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  
  # reshape into X=t and Y=t+1
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  print(testX.shape)
  print(testX[0])
  print(testY)
  
  # reshape input to be [samples, number of variables, look_back] *convert time series into column
  trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
  
  # create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back)))	#shape：変数数、遡る時間数
  model.add(Dense(1))
  model.add(LeakyReLU(alpha=0.3))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=30, batch_size=50, verbose=2)
  
  # make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)
  pad_col = np.zeros(dataset.shape[1]-1)


    
  trainPredict = scaler.inverse_transform(pad_array(trainPredict))
  trainY = scaler.inverse_transform(pad_array(trainY))
  testPredict = scaler.inverse_transform(pad_array(testPredict))
  testY = scaler.inverse_transform(pad_array(testY))

  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))
  

  testPredictPlot = pandas.DataFrame(testPredict[:,0])
  testActualPlot = pandas.DataFrame(testY[:,0])
  
  # output accurancy
  ListPredict, ListTest, Result= [], [], []
  for i in range(len(testPredictPlot)-1):
    if testPredict[i+1,0] > testPredict[i,0]:
        ListPredict.append("up")
    else:
        ListPredict.append("down")
        
    if testY[i+1,0] > testY[i,0]:
        ListTest.append("up")
    else:
        ListTest.append("down")
        
  for i in range(len(ListPredict)):
      if ListPredict[i] == ListTest[i]:
          Result.append("True")
      else:
          Result.append("False")
  Accurancy = Result.count("True")/len(Result)
  print('Accurancy: %.2f' % (Accurancy)) 
      
  # plot baseline and predictions
  plt.plot(testPredictPlot, label = 'PredictStockPrices')
  plt.plot(testActualPlot, label = 'ActualStockPrices')
  plt.legend()
  plt.show()