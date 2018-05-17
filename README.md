# RNN_Trade_Prediction 

### Examples
- LSTM_nikkei225.py

  日経平均株価をLong short-term memory(LSTM)に入力して、学習させ、株価を予測する。

  ![gazou1](https://github.com/dekabaron/gazou/blob/master/img/LSTM_nikkei225.png) 

Test Score: 282.52 RMSE
<br />

- LSTM_nikkei225andGDP.py 

  日経平均株価に加えて、GDPをLSTMに入力して、学習させ、株価を予測する。
  
  ![gazou2](https://github.com/dekabaron/gazou/blob/master/img/LSTM_nikkei225andGDP.png) 

Test Score: 369.08 RMSE
<br />

- trade_exchange_updown_classification.py
  
  為替と様々な国の株価データを、ニューラルネットワークに入力して、翌日の株価が上がるか下がるか予測する。  

  ![gazou1](https://github.com/dekabaron/gazou/blob/master/img/gazou3.png) 
  
  Precision =  0.48520710059171596  
  Recall =  0.6074074074074074  
  F1 Score =  0.5394736842105263  
  Accuracy =  0.5070422535211268  

### Requirements
requirements.txtに記載。
