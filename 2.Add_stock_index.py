#https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib      -> talib설치 
import talib
import os
import csv
import pandas as pd

download_path = "./1.download_stock_data/"
after_add_stock_index = "./2.after_add_stock_index/"
skip_date = 33

def add_stock_index_to_csv(path):
	#sma14,21, ema14,21, BollingerBands, rsi14,21, macd, macd_signal   나중에 넣어서 실험해볼 지표(#환율, 국제유가(1달단위....))
	file = os.listdir(path)

	for index, i in enumerate(file):
		df = pd.read_csv(path+i)
		
		#del Adj Close
		del df['Adj Close']
		
		#add sma
		df['sma14'] = talib.SMA(df['Close'], 14) 
		df['sma21'] = talib.SMA(df['Close'], 21) # 1일 ~ 21일 총 21일치의 정보를 사용해서 21일째에 정보가 붙음 따라서, 첫 1~20일은 정보가 없음. 
		
		#add ema
		df['ema14'] = talib.EMA(df['Close'], 14) 
		df['ema21'] = talib.EMA(df['Close'], 21) 
		
		#BollingerBands
		up, mavg, down = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
		df['bollingr_up'] = up
		df['bollinger_mavg'] = mavg
		df['bollinger_down'] = down

		#rsi14
		df['rsi14'] = talib.RSI(df['Close'], 14)
		df['rsi21'] = talib.RSI(df['Close'], 21)

		#macd
		macd, macd_signal, _ = talib.MACD(df['Close'], 12, 26, 9)
		df['macd'] = macd # 33개의 정보가 None임.
		df['macd_signal'] = macd_signal # 33개의 정보가 None임. 따라서 33개 줄은 제거해야됨. 
		
		#make csv
		df[skip_date:].to_csv(after_add_stock_index+i, index=False)#df[33:]는 앞 33개 줄 제거(title 제외),  index=False는 첫번째열에 넘버링 안붙게함.
	
		print(index+1, '/', 200, ' -> ', i, '지표 추가 완료')



if not os.path.isdir(after_add_stock_index):
	os.mkdir(after_add_stock_index)
	add_stock_index_to_csv(download_path)
	
else:
	print("이미 폴더가 존재하므로 종료합니다.")
