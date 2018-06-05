# TensorFlow-Stock_price_forecasting

## kospi200.csv
    * kospi200 종목들의 종목 코드 및 회사명(2018-06-01 기준)


## 1. Download_kospi200_price_csv.py
    * kospi200 종목들의 주가 정보 수집. (2010-01-01 ~ )
    * output file path : 1.download_stock_data
    
## 2. Add_stock_index.py
    * 수집한 kospi200 종목들에 sma14-21, ema14-21, BollingerBands, rsi14-21, macd, macd_signal 정보 추가.
    * input file path : 1.download_stock_data
    * output file path : 2.after_add_stock_index  
    
## 3. Data_preprocess.py
    * 데이터들을 날짜별로 20-gram으로 묶고 같은 의미를 갖는 column 단위로 데이터 정규화한 후 1row로 flatten.
    * output
        * 회사, 기준 날짜, 전처리 데이터, 기준 날짜의 종가, 기준 날짜부터 3일간의 종가/기준 날짜의 종가
            * 1 + 1 + (320=16*20) + 1 + 3 = 326 size.
    * 딥러닝의 입력 : 전처리된 데이터
    * 딥러닝의 타겟 : 기준 날짜부터 3일간의 종가/기준 날짜의 종가
    * 추론 : 예측된 output * 기준 날짜의 종가
    * input file path : 2.after_add_stock_index
    * output file path : 3.after_preprocess
    
## 4. Split_dataset.py    
    * 데이터를 학습, 검증, 테스트로 분할 : 0.7 : 0.15 : 0.15
    * input file path : 3.after_preprocess
    * output file path : 4.split_dataset

## 5. Deep_learning
    * Bi_LSTM_average_encoder_output.py  
    * Bi_LSTM_luong_attention.py
    * Bi_LSTM.py
        * image 폴더를 보면 위 3가지 모델의 오차( (실제 종가 - 예측 종가 / 실제 종가) * 100 )의 분포는 비슷함.
        * 즉 아무 모델이나 사용해도 결과는 비슷.