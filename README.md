# TensorFlow-Stock_price_forecasting

## kospi200.csv
    * kospi200 종목들의 종목 코드 및 회사명(2018-06-01 기준)


## 1.Download_kospi200_price_csv.py
    * kospi200 종목들의 주가 정보 수집. (2010-01-01 ~ 현재)
    * output file path : 1.download_stock_data
    
## 2.Add_stock_index.py
    * 수집한 kospi200 종목들에 sma14-21, ema14-21, BollingerBands, rsi14-21, macd, macd_signal 정보 추가.
    * input file path : 1.download_stock_data
    * output file path : 2.after_add_stock_index  
    
## 3.Data_preprocess.py
    * 데이터들을 날짜별로 20-gram으로 묶고 같은 의미를 갖는 column 단위로 데이터 정규화한 후 1row로 flatten.
    * output : 회사(1), 기준날짜(1), 전처리된 데이터(320=16*20), 기준close(1), 3일치의 target close/기준close/(1.3**3)(3). == 326 size
        * (1.3**3) : 주가가 하루에 30%이상 상승/하락 할 수 없는데, 3일 연속의 경우 1.3**3 이상 등락 불가능.
        * 따라서 1.3**3으로 나눠주면 0~1의 범위를 갖게 된다. => 크로스엔트로피와 시그모이드 사용 가능.
    * 딥러닝의 입력 : 전처리된 데이터
    * 딥러닝의 타겟 : 3일치의 target close/기준close
    * 추론 : 예측된 output * 기준close * (1.3**3)
    * input file path : 2.after_add_stock_index
    * output file path : 3.after_preprocess
    
## 4.Split_dataset.py    
    * 데이터를 학습, 검증, 테스트로 분할 : 0.7 : 0.15 : 0.15
    * total : 367739
        * train_set(70.0) : 257377
        * vali_set(15.0) : 55097
        * test_set(15.0) : 55265
    * input file path : 3.after_preprocess
    * output file path : 4.split_dataset
