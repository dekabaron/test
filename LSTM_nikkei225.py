#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:34:55 2018

@author: baron
"""
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error

look_back = 2
in_out_neurons = 1
hidden_neurons = 100

def load_data(data, n_prev):
  X, Y = [], []
  for i in range(len(data) - n_prev):
      # データフレーム型からarrya append で要素の追加
    X.append(data.iloc[i:(i+n_prev)].as_matrix()) #n_prev日の文だけデータを作成
    Y.append(data.iloc[i+n_prev].as_matrix()) #予想する日のデータの作成
  retX = np.array(X) 
  retY = np.array(Y) 
  return retX, retY



if __name__ == "__main__":

  # データ準備
  data = None
  data_ = pandas.read_csv('csv/nikkei225.csv',encoding="shift-jis")
  data = data_
  data.columns = ['date','open','high','low','close']
  data['date'] =pandas.to_datetime(data['date'],format='%Y/%m/%d')


  # 終値のデータを標準化
  close_max = max(data['close'])
  close_min = min(data['close'])
  
  data['close'] = (data['close']- close_min) / (close_max - close_min) #スケール化
  data = data.sort_values(by='date') 
  data = data.reset_index(drop=True)
  data = data.loc[:, ['date', 'close']] #dateとcloseの列のみの値を抽出

  # 2割をテストデータへ
  train_size = int(len(data) * 0.95)
  test_size = len(data) - train_size
  x_train, y_train = load_data(data[['close']].iloc[0:train_size], look_back)
  x_test, y_test = load_data(data[['close']].iloc[train_size:], look_back)
  
  model = Sequential()
  model.add(LSTM(hidden_neurons, \
          batch_input_shape=(None, look_back, in_out_neurons), \
          return_sequences=False))
  model.add(Dense(in_out_neurons))
  model.add(Activation("linear"))
  model.compile(loss="mape", optimizer="adam")
  model.fit(x_train, y_train, batch_size=50, epochs=10) #学習


  
  predicted = []
  predicted = model.predict(x_test)
 

  """
  #1日ごとに予測を行う
  Z = x_train[-1:]  # trainデータの一番最後を切り出し
  
  for i in range(test_size - look_back):
    z_ = Z[-1:]
    y_ = model.predict(z_) #翌日の株価を予測
    sequence_ = np.concatenate(
        (z_.reshape(look_back, in_out_neurons)[1:], y_),
        axis=0).reshape(1, look_back, in_out_neurons) 
    Z = np.append(Z, sequence_, axis=0) #予測した株価をlook_backに加える
    predicted.append(y_.reshape(-1)) 
  """
  
  
  predicted = np.array(predicted)
  predicted = predicted * (close_max - close_min) + close_min # 正規化を元に戻す
  y_test = y_test * (close_max - close_min) + close_min # 正規化を元に戻す
  
  # calculate root mean squared error
  testScore = math.sqrt(mean_squared_error(y_test[:,0], predicted[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))
  
  result = pandas.DataFrame(predicted)
  result.columns = ['predict']
  result['actual'] = y_test
  #result.to_csv("predict.csv")
  result.plot()
  plt.show()