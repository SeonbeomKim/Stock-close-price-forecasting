#데이터들을 20일치씩 20-gram으로 묶고
# 각 column별로 정규화 실시.
#20일치 정규화된 데이터는 한줄로 펴서 나중에 읽어오기 편하게 함.

import csv
import os
import numpy as np
import pandas as pd

after_add_stock_index = "./2.after_add_stock_index/"
after_preprocess = "./3.after_preprocess/"

gram = 20
forecasting_days = 3 #예측할 날짜 => 1일뒤, 2일뒤, 3일뒤 예측.
delta = 0.00001 # 0으로 나누는것 방지
skip_date = 33 #볼륨이 0이여서 제외한 마지막 데이터부터 33일간도 데이터 제외. macd, macd_signal 데이터가 34일차부터 생성되기 때문임.


def write_csv(data, file_name):
	with open(after_preprocess+file_name, 'w', newline='') as o:
		wr = csv.writer(o)

		for i in data:
			wr.writerow(i)


def normalize(column): #입력은 1차원으로 펴져있음. 이것을 정규분포 공식으로 정규화
	column = (column - np.mean(column)) / (np.std(column) + delta)
	column = np.round(column.astype(np.float32), 12) # 12자리에서 반올림.

	return column #1차원 리스트



# 액면 분할, 액면 병합, 기타 사유로 거래량(Volumn)이 0인 데이터가 20gram 내에 포함 되어 있는 경우에는 데이터 제외.
# ex) 한국타이어월드 2012-08-30 ~ 2012-10-02
# 마지막으로 제외한 데이터부터 33일간도 데이터 제외. macd, macd_signal 데이터가 34일차부터 생성되기 때문임.
# 마지막으로 제외한 데이터부터 33일간은 0은 아니지만 잘못된 값들이 들어가있음.
def check_abnormal_data(gram_data):
	volumn_column = gram_data[:, 5].astype(np.int32)
	if 0 in volumn_column: # 거래량이 0인것이 데이터에 포함되어 있는 경우
		return 0 #0이면 제외하겠다.
	else: # 
		return 1 # 1이면 포함하겠다.



# (회사), (기준날짜),  (날짜당 16개 컬럼. 20*16개 데이터), (기준close), (target close 3개) = 326개 사이즈 아웃풋냄.
def gram_normalization_and_flatten(gram_data, company): #gram 데이터를 컬럼별로 노멀라이즈하고 1자로 평평하게 펴줌.
	column_len = len(gram_data[0])

	date = gram_data[-1-forecasting_days][0] #기준 날짜.
	date_close = gram_data[-1-forecasting_days][4] #date 종가, 정규화 안함(나중에 테스트 체크용)
	target = gram_data[-forecasting_days:, 4].astype(np.float32) #date 이후 3일간의 종가
	target = target/date_close 
	
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




def preprocess(path):
	file = os.listdir(path)

	#모든 파일에 대해서.
	for index, i in enumerate(file):
		df = pd.read_csv(path+i, engine='python') #python 3.6에서는 engine을 python으로 지정 안해주면 파일명이 한글인 경우 실행 안됨. 3.5는 필요 없음.
	
		#파일당	
		preprocessed_data = []
		company = i.split(".")[0]
		skip = 0

		for k in range(len(df)):
			#gram(20)+ 다음3일치 sliding window 방식으로 읽음.
			gram_data = np.array(df[k:k+gram+forecasting_days]) # gram개 + 다음3일치의 close 가격 알아야하므로 gram+3개 뽑음.
			
			#gram(20)+ 다음3일치가 아닌경우 종료.
			if len(gram_data) != gram+forecasting_days:
				break


			#column별로 정규화
			if check_abnormal_data(gram_data) == 1: # 거래량이 0인 데이터가 없는 경우.
				if skip == 0: # 아무 문제 없는 데이터.
					normalized_result = gram_normalization_and_flatten(gram_data, company)
					preprocessed_data.append(normalized_result)
				else:  # skip이 0이 아닌 경우(거래량이 0인게 나왔어서 33줄을 버려야 하는 경우)에는 데이터 처리 안함(=제외)
					skip -=  1
			

			else: #거래량이 0인 데이터가 있는 경우, 거래량이 0인게 여러번 연속적으로 나오더라도 마지막 0 나오는 데이터 기준으로 skip이 33으로 세팅됨.
				skip = skip_date # 33

		
		# 회사명 깨짐
		#df = pd.DataFrame(preprocessed_data) 
		#df.to_csv(after_preprocess+i, index=False, header=False, encoding='utf-8')
		
		# 전처리된 데이터 생성 회사명 안깨짐.		
		write_csv(preprocessed_data, i)
		print(index+1, '/', 200, ' -> ', i, '전처리 완료')



def test_code(): # 한국타이어월드 데이터에서 거래량 0이였던것들 다 포함 안되고, 마지막 0 데이터 이후로 33개 버리고, 그 후부터 다시 데이터 수집 진행됨.
	df = pd.read_csv(after_add_stock_index+'한국타이어월드와이드.csv', engine='python') #engine을 python으로 지정 안해주면 한글 파일 실행 안됨.
	
	#파일당	
	preprocessed_data = []
	company = '한국타이어월드와이드'
	skip = 0

	for k in range(len(df)):
		#gram(20)+ 다음3일치 sliding window 방식으로 읽음.
		gram_data = np.array(df[k:k+gram+forecasting_days]) # gram개 + 다음3일치의 close 가격 알아야하므로 gram+3개 뽑음.
		
		#gram(20)+ 다음3일치가 아닌경우 종료.
		if len(gram_data) != gram+forecasting_days:
			break


		#column별로 정규화
		if check_abnormal_data(gram_data) == 1: # 거래량이 0인 데이터가 없는 경우.
			if skip == 0: # 아무 문제 없는 데이터.
				normalized_result = gram_normalization_and_flatten(gram_data, company)
				preprocessed_data.append(normalized_result)
			else:  # skip이 0이 아닌 경우(거래량이 0인게 나왔어서 33줄을 버려야 하는 경우)에는 데이터 처리 안함(=제외)
				skip -=  1
		

		else: #거래량이 0인 데이터가 있는 경우, 거래량이 0인게 여러번 연속적으로 나오더라도 마지막 0 나오는 데이터 기준으로 skip이 33으로 세팅됨.
			skip = skip_date # 33


	with open('./'+'한국타이어월드와이드.csv', 'w', newline='') as o:
		wr = csv.writer(o)

		for i in preprocessed_data:
			wr.writerow(i)





if not os.path.isdir(after_preprocess):
	os.mkdir(after_preprocess)
	preprocess(after_add_stock_index)
	print("가장 큰 target: ",global_max, '회사 : ', global_comp, "날짜 : ", global_date)

else:
	print("이미 폴더가 존재하므로 종료합니다.")


#testcode # 이상데이터 잘 처리 됐는지 확인용
#test_code()

