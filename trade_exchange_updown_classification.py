# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import datetime
import urllib
from os import path
import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
import tensorflow as tf
import matplotlib.pyplot as plt


DAYS_BACK = 7
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する
STOCK_MARKET_INDEXES = [
    #['DOW', '^DJI'],
    ['FTSE', '^FTSE'],
    ['GDAXI', '^GDAXI'],
    ['HSI', '^HSI'],
    ['N225', '^N225'],
    #['NASDAQ', '^IXIC'],
    ['SP500', '^GSPC'],
    ['FCHI', '^FCHI'],
    #['ASX','^AORD'],
    #['SSEC', '000001.SS'],
]

EXCHANGES_LABEL = [
        'JPYUSD',
        'EURUSD',
        'GBPUSD',
        'HKDUSD',
        'AUDUSD',
        'CNYUSD'
        ]
STOCK_MARKET_INDEXES_LABEL = [indexes[0] for indexes in STOCK_MARKET_INDEXES]

ALL_LABEL = STOCK_MARKET_INDEXES_LABEL + EXCHANGES_LABEL

Dataset = namedtuple(
    'Dataset',
    'training_predictors training_classes test_predictors test_classes')
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data')

        
def load_indexes_dataframes(target_indexes):
    '''indexesSに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Returns:
        {indexesS[n]: pd.DataFrame()}
    '''

    datas = {indexes: load_indexes_dataframe(indexes)
            for indexes in STOCK_MARKET_INDEXES_LABEL}

    # 計算対象の日付に存在しないデータを削除する
    if REMOVE_NIL_DATE:
        target_indexes = datas[target_indexes].index
        for (indexes, data) in datas.items():
            for index in data.index:
                if not index in target_indexes:
                    datas[indexes] = datas[indexes].drop(index)

    return datas
        
def load_indexes_dataframe(indexes):
    '''indexesに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Args:
        indexes: 指標名
    Returns:
        pd.DataFrame()
    '''
    dataframe = pd.read_csv('csv4/{}.csv'.format(indexes),encoding="shift-jis")
    dataframe = dataframe.replace('null','nan') #文字列nullをnanに置き換える
    dataframe = dataframe.convert_objects(convert_numeric=True) #objectをfloatに変換
    dataframe = dataframe.dropna(axis = 0) #nanを削除
    return dataframe.reset_index(drop=True) 
    
def get_using_indexes_data(dataframes):
    '''各指標の必要なカラムをまとめて1つのDataFrameに詰める。
    Args:
        dataframes: {key: pd.DataFrame()}
    Returns:
        pd.DataFrame()
    '''
    using_data = pd.DataFrame()
    for indexes, dataframe in dataframes.items():
        using_data['{}_OPEN'.format(indexes)] = dataframe['Open']
        using_data['{}_CLOSE'.format(indexes)] = dataframe['Close']
    using_data = using_data.fillna(method='ffill') #前日のデータで補完
    return using_data


def get_indexes_log_return_data(indexes_using_data):
    '''各指標について、終値を1日前との比率の対数をとって正規化する。
    Args:
        indexes_using_data: pd.DataFrame()
        exchanges_using_data: pd.DataFrame()
    Returns:
        pd.DataFrame()
    '''

    indexes_log_return_data = pd.DataFrame()
    for (indexes, _) in STOCK_MARKET_INDEXES:
        open_column = '{}_OPEN'.format(indexes)
        close_column = '{}_CLOSE'.format(indexes)
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出　（当日終値/当日始値のほうが良いのではないか）
        # 終値が前日よりも上がっていればプラス、下がっていればマイナスになる
        indexes_log_return_data['{}_CLOSE_RATE'.format(indexes)] = np.log(indexes_using_data[close_column]/indexes_using_data[close_column].shift())
        #log_return_data['{}_CLOSE_RATE'.format(indexes)] = np.log(using_data[close_column]/using_data[open_column])
        # その日の終値 >= 始値 なら1。それ意外は0
        indexes_log_return_data['{}_RESULT'.format(indexes)] = list(map(int, indexes_using_data[close_column] >= indexes_using_data[open_column]))

    return indexes_log_return_data

def get_exchanges_log_return_data(exchanges_using_data):
    exchanges_log_return_data = pd.DataFrame()    
    for (exchange) in EXCHANGES_LABEL:
        close_column = '{}_CLOSE'.format(exchange)
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出　（当日終値/当日始値のほうが良いのではないか）
        # 終値が前日よりも上がっていればプラス、下がっていればマイナスになる
        exchanges_log_return_data['{}_CLOSE_RATE'.format(exchange)] = np.log(exchanges_using_data[close_column]/exchanges_using_data[close_column].shift())
        
    return exchanges_log_return_data


def build_training_exchanges_data(log_return_data,  max_days_back=DAYS_BACK, use_subset=None):
    '''学習データを作る。分類クラスは、target_indexes指標の終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。
    Args:
        log_return_data: pd.DataFrame()
        target_indexes: 学習目標とする指標名
        max_days_back: 何日前までの終値を学習データに含めるか
        # 終値 >= 始値 なら1。それ意外は0
        use_subset (float): 短時間で動作を確認したい時用: log_return_dataのうち一部だけを学習データに含める
    Returns:
        pd.DataFrame()
    '''

    columns = []
    # 各指標のカラム名を追加
    for colname, _, _ in iter_exchanges_days_back(max_days_back):
        columns.append(colname)

    '''
    columns には計算対象の positive, negative と各指標の日数分のラベルが含まれる
    例：[
        'JPYUSD_1',
        'JPYUSD_2',
        'JPYUSD_3',
            ・
            ・
            ・        
    ]
    計算対象がSP500の場合、全て前日のデータを使う
    '''

    # データ数をもとめる
    max_index = len(log_return_data)
    if use_subset is not None:
        # データを少なくしたいとき
        max_index = int(max_index * use_subset)

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)
    for i in range(max_days_back + 10, max_index):
        # 先頭のデータを含めるとなぜか上手くいかないので max_days_back + 10 で少し省く
        values = {}
        # 学習データを入れる
        for colname, exchange, days_back in iter_exchanges_days_back(max_days_back):
            values[colname] = log_return_data['{}_CLOSE_RATE'.format(exchange)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True) #'positive','negative','GDAXI0','GDAXI1'.....'SSEC1','SSEC2'
    return training_test_data

def iter_exchanges_days_back(max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    for exchange in EXCHANGES_LABEL:
        start_days_back = 1 
        end_days_back = start_days_back + max_days_back
        for days_back in range(start_days_back, end_days_back):
            colname = '{}_{}'.format(exchange, days_back)
            yield colname, exchange, days_back
            
def build_training_indexes_data(log_return_data, target_indexes, max_days_back=DAYS_BACK, use_subset=None):
    '''学習データを作る。分類クラスは、target_indexes指標の終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。
    Args:
        log_return_data: pd.DataFrame()
        target_indexes: 学習目標とする指標名
        max_days_back: 何日前までの終値を学習データに含めるか
        # 終値 >= 始値 なら1。それ意外は0
        use_subset (float): 短時間で動作を確認したい時用: log_return_dataのうち一部だけを学習データに含める
    Returns:
        pd.DataFrame()
    '''

    columns = ['positive', 'negative']

    # 「上がる」「下がる」の結果を
    log_return_data['positive'] = 0
    positive_indices = op.eq(log_return_data['{}_RESULT'.format(target_indexes)], 1) #positive_indicesにtarget_indexesのResultをboolean型で代入
    log_return_data.ix[positive_indices, 'positive'] = 1 #log_return_data'positice'にpositive_indicesを代入
    log_return_data['negative'] = 0
    negative_indices = op.eq(log_return_data['{}_RESULT'.format(target_indexes)], 0)
    log_return_data.ix[negative_indices, 'negative'] = 1

    num_categories = len(columns)

    # 各指標のカラム名を追加
    for colname, _, _ in iter_indexes_days_back(target_indexes, max_days_back):
        columns.append(colname)

    '''
    columns には計算対象の positive, negative と各指標の日数分のラベルが含まれる
    例：[
        'positive',
        'negative',
        'DOW_0',
        'DOW_1',
        'DOW_2',
        'FTSE_0',
        'FTSE_1',
        'FTSE_2',
        'GDAXI_0',
        'GDAXI_1',
        'GDAXI_2',
        'HSI_0',
        'HSI_1',
        'HSI_2',
        'N225_0',
        'N225_1',
        'N225_2',
        'NASDAQ_0',
        'NASDAQ_1',
        'NASDAQ_2',
        'SP500_1',
        'SP500_2',
        'SP500_3',
        'SSEC_0',
        'SSEC_1',
        'SSEC_2'
    ]
    計算対象の SP500 だけ当日のデータを含めたらダメなので1〜3が入る
    '''

    # データ数をもとめる
    max_index = len(log_return_data)
    if use_subset is not None:
        # データを少なくしたいとき
        max_index = int(max_index * use_subset)

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)
    for i in range(max_days_back + 10, max_index):
        # 先頭のデータを含めるとなぜか上手くいかないので max_days_back + 10 で少し省く
        values = {}
        # 「上がる」「下がる」の答を入れる
        values['positive'] = log_return_data['positive'].ix[i]
        values['negative'] = log_return_data['negative'].ix[i]
        # 学習データを入れる
        for colname, indexes, days_back in iter_indexes_days_back(target_indexes, max_days_back):
            values[colname] = log_return_data['{}_CLOSE_RATE'.format(indexes)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True) #'positive','negative','GDAXI0','GDAXI1'.....'SSEC1','SSEC2'
    return num_categories, training_test_data

def iter_indexes_days_back(target_indexes, max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    for indexes in STOCK_MARKET_INDEXES_LABEL:
        # SP500 の結果を予測するのに SP500 の当日の値が含まれてはいけないので１日づらす
        start_days_back = 1 if indexes == target_indexes else 0
        #start_days_back = 1 # N225 で行う場合は全て前日の指標を使うようにする
        end_days_back = start_days_back + max_days_back
        for days_back in range(start_days_back, end_days_back):
            colname = '{}_{}'.format(indexes, days_back)
            yield colname, indexes, days_back


def split_training_test_data(num_categories, training_test_data):
    '''学習データをトレーニング用とテスト用に分割する。
    '''
    # 先頭２つより後ろが学習データ
    predictors_tf = training_test_data[training_test_data.columns[num_categories:]]
    # 先頭２つが「上がる」「下がる」の答えデータ
    classes_tf = training_test_data[training_test_data.columns[:num_categories]]

    # 学習用とテスト用のデータサイズを求める
    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size

    # 古いデータ0.8を学習とし、新しいデータ0.2がテストとなる
    return Dataset(
        training_predictors=predictors_tf[:training_set_size],
        training_classes=classes_tf[:training_set_size],
        test_predictors=predictors_tf[training_set_size:],
        test_classes=classes_tf[training_set_size:],
    )

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    '''与えられたネットワークの正解率などを出力する。
    '''
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = session.run(
        [tp_op, tn_op, fp_op, fn_op],
        feed_dict
    )

    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(tp) + float(fn))

    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    if (float(tp) + float(fp)):
        precision = float(tp)/(float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
    else:
        precision = 0
        f1_score = 0

    print('Precision = ', precision) #適合率
    print('Recall = ', recall) #再現率
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy) #正解率
    print('tp = ',float(tp)) #上がると予想して、当たった数
    print('tn = ',float(tn)) #下がると予想して、下がった数
    print('fp = ',float(fp)) #上がると予想して、外れた数
    print('fn = ',float(fn)) #下がると予想して、外れた数
    
def lrelu(x, alpha = 0.1):
    return tf.maximum(alpha * x, x)

def smarter_network(dataset):
    #隠れ層２つを含む、４層のニューラルネットワーク
    n_in = 100
    n_out = 2
    n_hidden1 = 50
    n_hidden2 = 25
    #n_hidden2 = 100
    
    sess = tf.Session() 

    num_predictors = len(dataset.training_predictors.columns)
    num_classes = len(dataset.training_classes.columns)

    feature_data = tf.placeholder("float", [None, num_predictors]) #入力変数
    actual_classes = tf.placeholder("float", [None, num_classes]) #答え
    # 入力層ー隠れ層
    W0 = tf.Variable(tf.truncated_normal([(DAYS_BACK * len(ALL_LABEL)), n_in], stddev=0.0001)) #(shape:Days_Back * STOCK_MARKET_INDEXES) * 50 の切断正規分布に従うデータをを用意
    b0 = tf.Variable(tf.ones([n_in]))
    h0 = lrelu(tf.matmul(feature_data, W0) + b0) #　h0=f(W0・x+b0)
    h0_drop = tf.nn.dropout(h0,0.5) #drop_outを適用
    
    # 隠れ層ー隠れ層
    W1 = tf.Variable(tf.truncated_normal([n_in, n_hidden1], stddev=0.0001))
    b1 = tf.Variable(tf.ones([n_hidden1]))
    h1 = lrelu(tf.matmul(h0_drop, W1) + b1) # h2=f(W2・h1+b2)
    h1_drop = tf.nn.dropout(h1,0.5) 
    
    W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.0001))
    b2 = tf.Variable(tf.ones([n_hidden2]))
    h2 = lrelu(tf.matmul(h1_drop, W2) + b2) # h2=f(W2・h1+b2)
    h2_drop = tf.nn.dropout(h2,0.5) 
    
    
    # 隠れ層ー出力層
    W3 = tf.Variable(tf.truncated_normal([n_hidden2, n_out], stddev=0.0001))
    b3 = tf.Variable(tf.ones([n_out]))
    model = tf.nn.softmax(tf.matmul(h2_drop, W3) + b3) # y=g(W3・h2＋b3)
    cross_entropy= -tf.reduce_sum(actual_classes*tf.log(model))

    training_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess.run(init)

    return Environ(
        sess=sess,
        model=model,
        actual_classes=actual_classes,
        training_step=training_step,
        dataset=dataset,
        feature_data=feature_data,
    )
    
def train(env, steps=30000, checkin_interval=5000):
    '''学習をsteps回おこなう。
    '''
    correct_prediction = tf.equal(
        tf.argmax(env.model, 1),
        tf.argmax(env.actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    for i in range(1, 1 + steps):
        env.sess.run(
            env.training_step,
            feed_dict=feed_dict(env, test=False),
        )
        if i % checkin_interval == 0:
            print(i, env.sess.run(
                accuracy,
                feed_dict=feed_dict(env, test=False),
            ))

    tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))


def feed_dict(env, test=False):
    '''学習/テストに使うデータを生成する。
    '''
    prefix = 'test' if test else 'training'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix)) #dataset型のobjectから入力データを抽出して代入
    classes = getattr(env.dataset, '{}_classes'.format(prefix)) #dataset型のobjectから答えのデータを抽出して代入
    return {
        env.feature_data: predictors.values,
        env.actual_classes: classes.values.reshape(len(classes.values), len(classes.columns))
    }
    
def close_data(dataframe):
    '''各指標の終値をまとめて1つのDataFrameに詰める。
    Args:
        dataframes:{}_OPEN and {}_CLOSE {}はEXCHSNGES_LABELで定義されている
    Returns:
        pd.DataFrame()
    '''
    close_data = pd.DataFrame()
    for indexes in STOCK_MARKET_INDEXES_LABEL:
        close_data['{}_CLOSE'.format(indexes)] = dataframe['{}_CLOSE'.format(indexes)]
    return close_data

def analysis_data(close_data):
    '''データの相関について分析する
    Args:
        dataframes:
    '''
    data_scaled = pd.DataFrame()
    for i in range(len(close_data.columns)):
        data_scaled[close_data.columns[i]] = close_data.iloc[:,i]/max(close_data.iloc[:,i])
    data_scaled.plot(figsize=(13,13))
    
    fig1 = plt.figure()
    fig1.set_figwidth(13)
    fig1.set_figheight(13)
    for i in range(len(data_scaled.columns)):
        autocorrelation_plot(data_scaled.iloc[:,i], label='{}'.format(data_scaled.columns[i]))
    scatter_matrix(data_scaled,diagonal='kde',figsize=(13,13))
    print(data_scaled.corr())
    
def load_exchanges_dataframes():
    '''exchangesに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Returns:
        {EXCHANGES_LABEL[n]: pd.DataFrame()}
    '''

    datas = {exchange: load_exchange_dataframe(exchange)
            for exchange in EXCHANGES_LABEL}

    return datas
        
def load_exchange_dataframe(exchange):
    '''indexesに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Args:
        exchange: 指標名
    Returns:
        pd.DataFrame()
    '''
    dataframe = pd.read_csv('EXCHANGES/{}.csv'.format(exchange),encoding="shift-jis")
    dataframe = dataframe.replace('null','nan') #文字列nullをnanに置き換える
    dataframe = dataframe.convert_objects(convert_numeric=True) #objectをfloatに変換
    dataframe = dataframe.dropna(axis = 0) #nanを削除
    return dataframe.reset_index(drop=True) 

def reshape_exchanges_data(exchanges_data,target_indexes):
    '''target_indexのindexのサイズに合わせる
    Args:
        exchanges_data,index
    Returns:
        pd.DataFrame()
    '''
    if REMOVE_NIL_DATE:
        for (indexes, exchange_data) in exchanges_data.items():
            for index in exchange_data.index:
                if not index in target_indexes:
                    exchanges_data[indexes] = exchanges_data[indexes].drop(index)

    return exchanges_data

def get_using_exchanges_data(dataframes):
    '''各指標の必要なカラムをまとめて1つのDataFrameに詰める。
    Args:
        dataframes: {key: pd.DataFrame()}
    Returns:
        pd.DataFrame()
    '''
    using_data = pd.DataFrame()
    for exchange, dataframe in dataframes.items():
        using_data['{}_CLOSE'.format(exchange)] = dataframe['Close']
    using_data = using_data.fillna(method='ffill') #前日のデータで補完
    return using_data


    
if __name__ == '__main__':
    target_indexes = 'SP500'
    steps = 100000
    checkin_interval = int(steps / 10)
    
    print('csvファイルの株価指標データをデータフレームに格納する')
    indexes_all_data = load_indexes_dataframes(target_indexes)
    
    print('終値と始め値を取得')
    indexes_using_data = get_using_indexes_data(indexes_all_data)
    
    print('csvファイルの為替データをデータフレームに格納する')
    exchanges_all_data = load_exchanges_dataframes()
    
    print('為替データを株価指標データのサイズに合わせる')
    reshape_exchanges_data = reshape_exchanges_data(exchanges_all_data,indexes_using_data.index)
    print('終値を取得')
    exchanges_using_data = get_using_exchanges_data(reshape_exchanges_data)
    
    print('使うデータの分析')
    #indexes_close_data = close_data(indexes_using_data)
    #all_close_data = pd.concat([indexes_close_data,exchanges_using_data],axis=1)
    #analysis_data(all_close_data)
    print('データを学習に使える形式に正規化')
    indexes_log_return_data = get_indexes_log_return_data(indexes_using_data)
    exchanges_log_return_data= get_exchanges_log_return_data(exchanges_using_data)
    
    print('答と学習データを作る')
    num_categories, training_test_data1 = build_training_indexes_data(indexes_log_return_data, target_indexes)
    training_test_data2 = build_training_exchanges_data(exchanges_log_return_data)
    print('学習データをトレーニング用とテスト用に分割する')
    training_test_data = pd.concat([training_test_data1,training_test_data2],axis=1)
    dataset = split_training_test_data(num_categories, training_test_data)

    print('ニューラルネットワークモデルの構築')
    env = smarter_network(dataset)
    
    print('学習') 
    train(env, steps=steps, checkin_interval=checkin_interval)
        
        