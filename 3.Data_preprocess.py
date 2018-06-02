#데이터들을 20일치씩 20-gram으로 묶고
# 각 column별로 정규화 실시.
#20일치 정규화된 데이터는 한줄로 펴서 나중에 읽어오기 편하게 함.

import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #pip install scikit-learn, scipy
import warnings
warnings.filterwarnings("ignore") #MinMaxScaling 할때 워닝 안뜨게함.

after_add_stock_index = "./2.after_add_stock_index/"
after_preprocess = "./3.after_preprocess/"

gram = 20
forecasting_days = 3 #예측할 날짜 => 1일뒤, 2일뒤, 3일뒤 예측.
scaler = MinMaxScaler(feature_range=(0, 1)) #scaler 선언 데이터 분포를 -1~1 범위로 맞추겠다.

#규제때문에 하루만에 30% 이상 오를 수 없음. 3일연속이라면 1.3**3 이상 오를 수 없음.  
#따라서 기준1일뒤 종가/기준날 종가, 2일뒤 종가/기준날 종가, 3일뒤 종가/기준날 종가를 이 값으로 나눠주면 0~1 사이로 고정됨.
#즉 크로스엔트로피랑 시그모이드 쓸 수 있음.
price_upper = 1.3**3  #나중에 해보고 잘 안되면 지울것.


def normalize(column): #입력은 1차원으로 펴져있는데 이걸 다시 컬럼화해서 정규화하고 다시 1자로 펴줌.
	column = np.reshape(column, (-1, 1))
	column = scaler.fit_transform(column) #컬럼별로 노말라이제이션
	column = np.round(column, 12) # 12자리에서 반올림.
	column = np.reshape(column, (-1)) #다시 1자로 펴줌.

	return column #1차원 리스트



# (회사), (기준날짜),  (날짜당 16개 컬럼. 20*16개 데이터), (기준close), (target close 3개) = 326개 사이즈 아웃풋냄.
def gram_normalization_and_flatten(gram_data, company): #gram 데이터를 컬럼별로 노멀라이즈하고 1자로 평평하게 펴줌.
	column_len = len(gram_data[0])

	date = gram_data[-1-forecasting_days][0] #기준 날짜.
	date_close = gram_data[-1-forecasting_days][4] #date 종가, 정규화 안함(나중에 테스트 체크용)
	target = gram_data[-forecasting_days:, 4] #date 이후 3일간의 종가
	target = target/date_close 
	target = target / price_upper #나중에 해보고 잘 안되면 지울것.

	preprocessed_data = [company, date] #회사명, 기준 날짜.

	#컬럼별 정규화
	gram_data = gram_data[:gram] # gram+forecasting_days 중에 gram만 씀. 뒤 forecasting_days는 close만 필요한거니까 안씀.
	for column in range(1, column_len): # 0은 날짜. 정규화 할 필요가 없음.
		gram_data[:, column] = normalize(gram_data[:, column]) #1차원리스트가 리턴되어야 컬럼에 들어가짐.
	

	#flatten	
	flatten = np.reshape(gram_data[:, 1:], (-1)).tolist() #날짜 제외하고 정규화된 값들을 1자로 다 펴줌.
	preprocessed_data.extend(flatten) #flatten은 1차 리스트니까 extend
	preprocessed_data.append(date_close) # date_close는 float이니까 append
	preprocessed_data.extend(target.tolist()) # target은 1차원 넘파이니까 리스트로 바꾸고 extend

	return preprocessed_data # 326사이즈의 1차 리스트. 회사,기준날짜,데이터,기준close,target close 순서.



def write_csv(data, file_name):
	with open(after_preprocess+file_name, 'w', newline='') as o:
		wr = csv.writer(o)

		for i in data:
			wr.writerow(i)



def preprocess(path):
	file = os.listdir(path)

	#모든 파일에 대해서.
	for index, i in enumerate(file):
		df = pd.read_csv(path+i)
	
		#파일당	
		preprocessed_data = []
		company = i.split(".")[0]
		for k in range(len(df)):
			#gram(20)+ 다음3일치 sliding window 방식으로 읽음.
			gram_data = np.array(df[k:k+gram+forecasting_days]) # gram개 + 다음3일치의 close 가격 알아야하므로 gram+3개 뽑음.
			
			#gram(20)+ 다음3일치가 아닌경우 종료.
			if len(gram_data) != gram+forecasting_days:
				break

			#column별로 정규화
			normalized_result = gram_normalization_and_flatten(gram_data, company)
			preprocessed_data.append(normalized_result)
			
		
		# 회사명 깨짐
		#df = pd.DataFrame(preprocessed_data) 
		#df.to_csv(after_preprocess+i, index=False, header=False, encoding='utf-8')
		
		# 전처리된 데이터 생성 회사명 안깨짐.		
		write_csv(preprocessed_data, i)
		print(index+1, '/', 200, ' -> ', i, '전처리 완료')



if not os.path.isdir(after_preprocess):
	os.mkdir(after_preprocess)
	preprocess(after_add_stock_index)

else:
	print("이미 폴더가 존재하므로 종료합니다.")
