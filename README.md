# TensorFlow-Stock_price_forecasting

## 1.Download_kospi200_price_csv.py
    * kospi200 종목들의 주가 정보 수집. (2010-01-01 ~ 현재)
    * output file path : download_stock_data
    
## 2.add_stock_index.py
    * 수집한 kospi200 종목들에 sma14-21, ema14-21, BollingerBands, rsi14-21, macd, macd_signal 정보 추가.
    * input file path : download_stock_data
    * output file path : 2.after_add_stock_index  
    
## 3.data_preprocess.py
    * 데이터들을 날짜별로 20-gram으로 묶고 같은 의미를 갖는 column 단위로 데이터 정규화 실시. 
    * 최종적으로 전처리된 20-gram 데이터를 1row 로 flatten. 
    * output정보 : 회사, 기준날짜, 전처리된 데이터, 기준close, 3일치의 target close/기준close.
    * input file path : 2.after_add_stock_index
    * output file path : 3.after_preprocess
