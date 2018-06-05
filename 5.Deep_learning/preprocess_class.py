#실제 예측과, 실제가격과 예측가격 비교할때 필요한 전처리 코드 클래스.
# 1. ~ 3. 전처리 코드 합쳐둔것임. 필요한 부분에 맞게 수정해서 사용함.

from pandas_datareader import data #pip install git+https://github.com/pydata/pandas-datareader
import talib #https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib      -> talib설치 pip install 파일명
import numpy as np


# 실제 예측할때 쓸 코드
class for_inference:
	def __init__(self, symbol, company, start=None, end=None):
		self.symbol = symbol
		self.company = company
		self.start = start
		self.end = end

		self.gram = 20 #20일치 정보 주고 3일치 예측.
		self.forecasting_days = 3 #예측할 날짜 => 1일뒤, 2일뒤, 3일뒤 예측.
		self.delta = 0.00001 # 0으로 나누는것 방지
		self.skip_date = 33 #볼륨이 0이여서 제외한 마지막 데이터부터 33일간도 데이터 제외. macd, macd_signal 데이터가 34일차부터 생성되기 때문임.



	def get_stock_data(self): # 우리는 마지막 20개만 있으면 됨. 하지만 지표 계산에 33개가 더 필요하므로 53개 추출.
		symbol = self.symbol+'.KS' #야후 기업 코드 양식.
		df = data.get_data_yahoo(symbol, start=self.start, end=self.end)

		# 날짜를 0번인덱스로 할당해줌. csv로 저장했다가 불러오면 당연히 날짜가 0인데. 그냥 get_data한 경우에는 날짜는 인덱스지정불가
		df.reset_index(inplace=True,drop=False) 

		df = df[-self.gram - self.skip_date: ] # 33개는 지표다는데 사용되고 34번째 ~ 53까지 즉 20개로 추론함.
		check_zero_volumn = np.array(df)[:, 5].astype(np.int32) # 마지막 20+33 개에 거래량이 0인것이 포함되어 있는 경우 무슨 문제가 있는지 알 수 없으므로 처리안함.
		
		return df, 0 in check_zero_volumn #df, [True or false]



	def add_stock_index(self, df):
		
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

		return df[self.skip_date:] #df 자체는 33줄의 0 포함.



	def normalize(self, column): #입력은 1차원으로 펴져있음. 이것을 정규분포 공식으로 정규화
		column = (column - np.mean(column)) / (np.std(column) + self.delta)
		column = np.round(column.astype(np.float32), 12) # 12자리에서 반올림.

		return column #1차원 리스트



	# (회사), (기준날짜),  (날짜당 16개 컬럼. 20*16개 데이터), (기준close) = 323개 사이즈 아웃풋냄. 
	def gram_normalization_and_flatten(self, gram_data): #gram 데이터를 컬럼별로 노멀라이즈하고 1자로 평평하게 펴줌.
		column_len = len(gram_data[0])

		date = gram_data[-1][0] #기준 날짜.
		date_close = gram_data[-1][4] #date 종가, 정규화 안함(나중에 테스트 체크용)
		
		preprocessed_data = [self.company, date] #회사명, 기준 날짜 .

		#컬럼별 정규화
		for column in range(1, column_len): # 0은 날짜. 정규화 할 필요가 없음.
			gram_data[:, column] = self.normalize(gram_data[:, column]) #1차원리스트가 리턴되어야 컬럼에 들어가짐.
		

		#flatten	
		flatten = np.reshape(gram_data[:, 1:], (-1)).tolist() #날짜 제외하고 정규화된 값들을 1자로 다 펴줌.
		preprocessed_data.extend(flatten) #flatten은 1차 리스트니까 extend
		preprocessed_data.append(date_close) # date_close는 float이니까 append
		
		return preprocessed_data # 323사이즈의 1차 리스트. 회사,기준날짜,데이터,기준close 순서



	# 추론시에는 마지막 3일치 필요 없음.
	def preprocess(self, df):
		preprocessed_data = []

		for k in range(len(df)):
			#gram(20) sliding window 방식으로 읽음.
			gram_data = np.array(df[k:k + self.gram]) # gram개 정보 알아야하므로 gram개 뽑음.
			
			#gram(20) 아닌경우 종료.
			if len(gram_data) != self.gram:
				break
		
			normalized_result = self.gram_normalization_and_flatten(gram_data)
			preprocessed_data.append(normalized_result)
	
		return np.array(preprocessed_data)
		




# 주가 실제랑 예측이랑 겹쳐 그릴때만 쓸 코드
class for_draw_test(for_inference): #for_inference랑 비슷한 부분 많아서 상속시킴.

	def get_stock_data(self): # 주가 실제랑 예측이랑 겹쳐 그릴때만 쓸 코드. 전체 데이터에 0이 없어야 정상적인 확인 가능.
		symbol = self.symbol+'.KS' #야후 기업 코드 양식.
		df = data.get_data_yahoo(symbol, start=self.start, end=self.end)
		
		# 날짜를 0번인덱스로 할당해줌. csv로 저장했다가 불러오면 당연히 날짜가 0인데. 그냥 get_data한 경우에는 날짜는 인덱스지정불가
		df.reset_index(inplace=True,drop=False) 

		check_zero_volumn = np.array(df)[:, 5].astype(np.int32) #거래량이 0인것이 포함되어 있는 경우 무슨 문제가 있는지 알 수 없으므로 처리안함.

		return df, 0 in check_zero_volumn #df, [True or false]



	# (회사), (기준날짜),  (날짜당 16개 컬럼. 20*16개 데이터), (기준close), (target close 3개) = 326개 사이즈 아웃풋냄.
	def gram_normalization_and_flatten(self, gram_data): #gram 데이터를 컬럼별로 노멀라이즈하고 1자로 평평하게 펴줌.
		column_len = len(gram_data[0])

		date = gram_data[-1 - self.forecasting_days][0] #기준 날짜.
		date_close = gram_data[-1 - self.forecasting_days][4] #date 종가, 정규화 안함(나중에 테스트 체크용)
		target = gram_data[-self.forecasting_days:, 4].astype(np.float32) #date 이후 3일간의 종가
		target = target/date_close 
		
		preprocessed_data = [self.company, date] #회사명, 기준 날짜 .

		#컬럼별 정규화
		gram_data = gram_data[:self.gram] # gram+forecasting_days 중에 gram만 씀. 뒤 forecasting_days는 close만 필요한거니까 안씀.
		for column in range(1, column_len): # 0은 날짜. 정규화 할 필요가 없음.
			gram_data[:, column] = self.normalize(gram_data[:, column]) #1차원리스트가 리턴되어야 컬럼에 들어가짐.
		

		#flatten	
		flatten = np.reshape(gram_data[:, 1:], (-1)).tolist() #날짜 제외하고 정규화된 값들을 1자로 다 펴줌.
		preprocessed_data.extend(flatten) #flatten은 1차 리스트니까 extend
		preprocessed_data.append(date_close) # date_close는 float이니까 append
		preprocessed_data.extend(target.tolist()) # target은 1차원 넘파이니까 리스트로 바꾸고 extend

		return preprocessed_data # 326사이즈의 1차 리스트. 회사,기준날짜,데이터,기준close,target close 순서.



	def preprocess(self, df):
		preprocessed_data = []

		for k in range(len(df)):
			#gram(20)+ 다음3일치 sliding window 방식으로 읽음.
			gram_data = np.array(df[k:k + self.gram + self.forecasting_days]) # gram개 + 다음3일치의 close 가격 알아야하므로 gram+3개 뽑음.
			
			#gram(20)+ 다음3일치가 아닌경우 종료.
			if len(gram_data) != self.gram + self.forecasting_days:
				break
		
			normalized_result = self.gram_normalization_and_flatten(gram_data)
			preprocessed_data.append(normalized_result)
	
		return np.array(preprocessed_data)