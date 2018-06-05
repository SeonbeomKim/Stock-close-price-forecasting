#kospi200 종목들의 주가 데이터 수집 코드.

#pip install pandas-datareader로 설치한 후에 import 에러나, 실행 에러 나는 경우에는
#pip install git+https://github.com/pydata/pandas-datareader 설치.

from pandas_datareader import data
import csv
import os

download_path = "./1.download_stock_data/"
kospi200_path = './kospi200/kospi200.csv'
start = '2010-01-01' #데이터 시작 시점.

def download_stock_data(path):
	with open(path, 'r', newline='') as o:
		wr = csv.reader(o)
		for index, i in enumerate(wr): # 총 200개.
			symbol = i[0][3:9]+'.KS' #야후 기업 코드 양식.
			company = i[1] #회사명

			df = data.get_data_yahoo(symbol, start)
			df.to_csv(download_path+company+'.csv')

			print(index+1, '/', 200, ' -> ', symbol, company, '완료')

if not os.path.isdir(download_path):
	os.mkdir(download_path)
	download_stock_data(kospi200_path)
	
else:
	print("이미 폴더가 존재하므로 종료합니다.")

=======
#kospi200 종목들의 주가 데이터 수집 코드.

#pip install pandas-datareader로 설치한 후에 import 에러나, 실행 에러 나는 경우에는
#pip install git+https://github.com/pydata/pandas-datareader 설치.

from pandas_datareader import data
import csv
import os

download_path = "./1.download_stock_data/"
kospi200_path = './kospi200/kospi200.csv'
start = '2010-01-01' #데이터 시작 시점.

def download_stock_data(path):
	with open(path, 'r', newline='') as o:
		wr = csv.reader(o)
		for index, i in enumerate(wr): # 총 200개.
			symbol = i[0][3:9]+'.KS' #야후 기업 코드 양식.
			company = i[1] #회사명

			df = data.get_data_yahoo(symbol, start)
			df.to_csv(download_path+company+'.csv')

			print(index+1, '/', 200, ' -> ', symbol, company, '완료')

if not os.path.isdir(download_path):
	os.mkdir(download_path)
	download_stock_data(kospi200_path)
	
else:
	print("이미 폴더가 존재하므로 종료합니다.")

>>>>>>> fcdb2483f25d1b200c57673da68296fdf8b6c395
