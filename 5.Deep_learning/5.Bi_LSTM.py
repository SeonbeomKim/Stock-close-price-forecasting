import tensorflow as tf #version=1.4
import numpy as np
import csv
import pandas as pd
import os

tensorboard_path = './tensorboard_Bi_LSTM/'
saver_path = './saver_Bi_LSTM/'

train_path = "../4.split_dataset/train_set.csv"
vali_path = "../4.split_dataset/vali_set.csv"
test_path = "../4.split_dataset/test_set.csv"

train_rate = 0.00001
cell_num = 512
gram = 20 #20 gram
column = 16 #지표가 16개임. 
target_size = 3



class model:
	def __init__(self, sess):

		#placeholder
		self.X = tf.placeholder(tf.float32, [None, gram*column]) #batch
		self.Y = tf.placeholder(tf.float32, [None, target_size]) #batch
		self.X_reshape = tf.reshape(self.X, (-1, gram, column))

		
		#bi_lstm
		self.en_last_output = self.bidirectional_encoder(self.X_reshape) #bidirectional, en_output = batch, gram(20), cell_num*2	


		#prediction
		self.pred = self.prediction(self.en_last_output) # batch, target


		#optimizer
		self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.Y-self.pred)))
		self.optimizer = tf.train.AdamOptimizer(train_rate)
		self.minimize = self.optimizer.minimize(self.cost)


		#tensorboard
		self.train_loss = tf.placeholder(tf.float32, name='train_loss')
		self.vali_loss = tf.placeholder(tf.float32, name='vali_loss')
		self.test_mean_error_1 = tf.placeholder(tf.float32, name='test_mean_error_1') #테스트데이터 1일뒤 오차 평균
		self.test_mean_error_2 = tf.placeholder(tf.float32, name='test_mean_error_2') #테스트데이터 2일뒤 오차 평균
		self.test_mean_error_3 = tf.placeholder(tf.float32, name='test_mean_error_3') #테스트데이터 3일뒤 오차 평균

		self.train_summary = tf.summary.scalar("train_loss", self.train_loss)
		self.vali_summary = tf.summary.scalar("vali_loss", self.vali_loss)
		self.test_summary_1 = tf.summary.scalar("test_mean_error_1", self.test_mean_error_1)
		self.test_summary_2 = tf.summary.scalar("test_mean_error_2", self.test_mean_error_2)
		self.test_summary_3 = tf.summary.scalar("test_mean_error_3", self.test_mean_error_3)

		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


		#variable initialize
		sess.run(tf.global_variables_initializer())


		#saver
		self.saver = tf.train.Saver(max_to_keep=10000)



	def bidirectional_encoder(self, X_reshape):
		fw = tf.nn.rnn_cell.LSTMCell(cell_num) # cell must write here. not in def function
		bw = tf.nn.rnn_cell.LSTMCell(cell_num)

		((en_fw_val, en_bw_val), (en_fw_state, en_bw_state)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, X_reshape, dtype=tf.float32)
		en_state_h = tf.concat((en_fw_state.h, en_bw_state.h), 1)
	
		return en_state_h



	def decoder(self, batch_size, en_state):
		decoder_cell = tf.nn.rnn_cell.LSTMCell(cell_num*2) # encoder is bidirectional, so decoder_cell_num is 2*encoder_cell_num
		decoder_go = tf.ones([batch_size, 1, 1], dtype=tf.float32) # batch, 1, 1
		de_output, _ = tf.nn.dynamic_rnn(decoder_cell, decoder_go, dtype=tf.float32, initial_state = en_state)
		
		return de_output



	def prediction(self, concat_vector):
		W = tf.get_variable('w', shape = [cell_num*2, target_size], 
				initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.Variable(tf.constant(1.0, shape = [target_size]))	
		pred = tf.matmul(concat_vector, W) + bias

		return pred




def train(train_path, model):
	batch_size = 256
	loss = 0

	#csv 파일 용량이 크기 때문에 저장해서 쓰지 않고, pandas를 이용해서 필요한부분 batch_size 만큼씩만 읽어옴.
	for index, df in enumerate(pd.read_csv(train_path, chunksize=batch_size, iterator=True, encoding='euc-kr', header=None)):
		input_ = np.array(df)[:, 2:-4] #batch, gram*column #정규화했던 데이터만.
		target_ = np.array(df)[:, -3:] #batch, target_size #1~3 일 뒤의 종가
	
		train_loss, _ = sess.run([model.cost, model.minimize], {model.X:input_, model.Y:target_})
		loss += train_loss
		
	return loss / (index+1) # not exact : because totalset % batch_size != 0




def validation(vali_path, model):
	batch_size = 1024
	loss = 0
	
	for index, df in enumerate(pd.read_csv(vali_path, chunksize=batch_size, iterator=True, encoding='euc-kr', header=None)):
		input_ = np.array(df)[:, 2:-4] #batch, gram*column #정규화했던 데이터만.
		target_ = np.array(df)[:, -3:] #batch, target_size #1~3 일 뒤의 종가
	
		vali_loss = sess.run(model.cost, {model.X:input_, model.Y:target_}) 
		loss += vali_loss
	
	return loss / (index+1) # not exact : because totalset % batch_size != 0




def test(test_path, model):
	batch_size = 1024
	total_error = np.zeros(target_size)
	data_size = 0

	for index, df in enumerate(pd.read_csv(test_path, chunksize=batch_size, iterator=True, encoding='euc-kr', header=None)):
		input_ = np.array(df)[:, 2:-4] #batch, gram*column #정규화했던 데이터만.
		target_ = np.array(df)[:, -3:] #batch, target_size #1~3 일 뒤의 종가/기준종가/1.3**3
		date_close = np.reshape(np.array(df)[:, -4], (-1, 1)) # 기준 날짜의 종가. 곱하기 위해 형태맞춰줌.
		
		data_size += len(input_)

		#예측 종가. 딥러닝 아웃풋에 date_close 곱해주면 됨. => 전처리할때 date_close로 나눠줬으니 다시 곱해줘야함.
		pred_price = sess.run(model.pred, {model.X:input_}) 
		pred_price = np.array(pred_price) * date_close # 예측한 실제 가격,
		
		#실제 종가.
		real_price = target_ * date_close

		#오차 계산 절대값, %단위.
		error = np.abs((real_price - pred_price) / real_price * 100)
		error = np.sum(error, axis=0).astype(np.float32)
		total_error += error

	return total_error / data_size # 3 사이즈 리스트 리턴 [다음날 오차, 이틀 뒤 오차, 삼일 뒤 오차]




def run(train_path, vali_path, test_path, model, restore=-1):
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)

	#restore check
	if restore != -1:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	else:
		restore = 0

	#train,vali,test
	for epoch in range(restore + 1, 50):
		train_loss = train(train_path, model)
		vali_loss = validation(vali_path, model)
		test_error = test(test_path, model)
		print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, ' test_error : ', test_error)
		
		summary = sess.run(model.merged, {
							model.train_loss:train_loss,
							model.vali_loss:vali_loss,
							model.test_mean_error_1:test_error[0],
							model.test_mean_error_2:test_error[1],
							model.test_mean_error_3:test_error[2]
						}
					)
		model.writer.add_summary(summary, epoch)


		if epoch % 2 == 0:
			save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")




# 실제 20일간의 데이터로 다음 3일치 가격 예측코드
def inference(model, restore, symbol, company, start=None):
	from preprocess_class import for_inference #추론용 데이터 수집 지표 전처리 수행하는 클래스
	
	#restore
	model.saver.restore(sess, saver_path+str(restore)+".ckpt")

	#get data, 수집, 지표, 전처리 모든 과정 수행
	infer = for_inference(symbol, company, start=start) #실제론 이 날부터 33일 뒤 데이터 사용됨. 지표계산하느라 33일 쓰레기값.
	
	#데이터 53개 수집. (53개중 33개는 지표계산용)
	df, isZero = infer.get_stock_data() # 실제 예측할때.
	if isZero == True: #거래량이 0인 비정상 데이터가 존재하는 경우
		print("거래량이 0인 데이터가 존재합니다")
		return

	#데이터 전처리 완료(20일치의 데이터), 최종 1, 323 shape.   회사,기준 날짜,320데이터, 날짜종가.
	df = infer.add_stock_index(df)
	df = infer.preprocess(df)

	#딥러닝
	info_ = np.array(df)[:, :2] #company, date
	input_ = np.array(df)[:, 2:-1] #batch, gram*column #정규화했던 데이터만.
	date_close = np.reshape(np.array(df)[:, -1], (-1, 1)) # 기준 날짜의 종가. 곱하기 위해 형태맞춰줌.
	
	#예측 종가. 딥러닝 아웃풋에 date_close 곱해주면 됨. => 전처리할때 date_close로 나눠줬으니 다시 곱해줘야함.
	pred_price = sess.run(model.pred, {model.X:input_}) 
	pred_price = np.array(pred_price) * date_close # 예측한 실제 가격,
	
	return info_[0][0]+' '+ str(info_[0][1]) + ' : ' + str(date_close[0]) + ', 예측 가격[1, 2, 3 일 뒤] : ' + str(pred_price[0])





# image/price_graph 에 저장된 그림 생성 코드 (1일 뒤의 예측 가격끼리만 비교함. 2,3일은 안함 오차가 너무 큼.)
def draw_test_graph(model, restore, symbol, company, start=None): 
	import matplotlib.pyplot as plt #pip install matplotlib
	from preprocess_class import for_draw_test #그래프 그리기용 데이터 수집 지표 전처리 수행하는 클래스
	
	#restore
	model.saver.restore(sess, saver_path+str(restore)+".ckpt")

	#get data, 수집, 지표, 전처리 모든 과정 수행
	draw = for_draw_test(symbol, company, start=start)
	
	#start 날짜부터 전부 수집
	df, isZero = draw.get_stock_data()
	if isZero == True: #거래량이 0인 비정상 데이터가 존재하는 경우
		print("거래량이 0인 데이터가 존재합니다")
		return

	#데이터 전처리 완료(수집한것-33개 데이터), 최종 (수집-33, 326) shape.   회사,기준 날짜,320데이터, 날짜종가, 실제3일종가/날짜종가
	df = draw.add_stock_index(df)
	df = draw.preprocess(df)

	#딥러닝
	info_ = np.array(df)[:, :2] #company, date
	input_ = np.array(df)[:, 2:-4] #batch, gram*column #정규화했던 데이터만.
	target_ = np.array(df)[:, -3:] #batch, target_size #1~3 일 뒤의 종가/기준종가/1.3**3
	date_close = np.reshape(np.array(df)[:, -4], (-1, 1)) # 기준 날짜의 종가. 곱하기 위해 형태맞춰줌.
	
	#예측 종가. 딥러닝 아웃풋에 date_close 곱해주면 됨. => 전처리할때 date_close로 나눠줬으니 다시 곱해줘야함.
	pred_price = sess.run(model.pred, {model.X:input_}) 
	pred_price = np.array(pred_price) * date_close # 예측한 실제 가격,
	
	#실제 종가.
	real_price = target_ * date_close

	plt.plot(info_[:, 1], real_price[:, 0], label='real_price') # 날짜와 1일뒤 실제 종가
	plt.plot(info_[:, 1], pred_price[:, 0], label='pred_price')
	plt.title("Price forecast for the next day")
	plt.legend()
	plt.show()





# image/error/ 에러 분포 그림 생성하는 코드
def distribution(test_path, model, restore):
	#시각화
	import matplotlib.pyplot as plt #pip install matplotlib

	#restore
	model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	
	#딥러닝 오차 수집
	batch_size = 1024	
	see_distribution = []

	for index, df in enumerate(pd.read_csv(test_path, chunksize=batch_size, iterator=True, encoding='euc-kr', header=None)):
		input_ = np.array(df)[:, 2:-4] #batch, gram*column #정규화했던 데이터만.
		target_ = np.array(df)[:, -3:] #batch, target_size #1~3 일 뒤의 종가/기준종가/1.3**3
		date_close = np.reshape(np.array(df)[:, -4], (-1, 1)) # 기준 날짜의 종가. 곱하기 위해 형태맞춰줌.
		
		pred_price = sess.run(model.pred, {model.X:input_}) 
		pred_price = np.array(pred_price) * date_close
		
		#실제 종가.
		real_price = target_*date_close

		#오차 계산 절대값, %단위.
		error = np.abs((real_price - pred_price) / real_price * 100)
		
		#check distribution
		for i in error:
			see_distribution.append(i)
		
	#to numpy
	see_distribution = np.array(see_distribution, dtype=np.float32)

	#1일차 오차
	plt.subplot(1,3, 1) 
	n, bins, patches = plt.hist(see_distribution[:, 0], 20)
	plt.title('error distribution : after_1_day')
	plt.xlim([0,40]) #x축 범위 0~40
	plt.ylim([0, len(see_distribution)])

	#2일차 오차
	plt.subplot(1,3, 2) 
	n, bins, patches = plt.hist(see_distribution[:, 1], 20)
	plt.title('error distribution : after_2_days')
	plt.xlim([0,40])
	plt.ylim([0, len(see_distribution)])

	#3일차 오차
	plt.subplot(1,3, 3) 
	n, bins, patches = plt.hist(see_distribution[:, 2], 20)
	plt.title('error distribution : after_3_days')
	plt.xlim([0,40])
	plt.ylim([0, len(see_distribution)])

	#그리기
	plt.tight_layout()
	plt.legend()

	plt.show()



sess = tf.Session()
stock = model(sess)

#학습 검증 테스트
run(train_path, vali_path, test_path, stock, restore=-1)


#에러 분포 확인. like image/error/ 
#distribution(test_path, stock, 12) # 에러 분포 확인


#inference. 1~3일 뒤 가격 예측.
#result = inference(stock, 48, '089590', '제주항공', start='2017-11-28')
#print(result)


#실제 가격과 예측 가격 그래프 비교. like image/price_graph
#draw_test_graph(stock, 48, '089590', '제주항공', start='2017-11-28')
#draw_test_graph(stock, 48, '089590', '제주항공', start='2018-01-28')



