<<<<<<< HEAD
import csv
import numpy as np
import os

after_preprocess = "./3.after_preprocess/"
split_dataset_path = "./4.split_dataset/"

train_ratio = 0.7
vali_ratio = 0.15
test_ratio = 0.15

def data_read(path):
	with open(path, 'r', newline='') as f:
		re = csv.reader(f)
		return np.array(list(re))


def split_data(path):
	file = os.listdir(path)

	num_train_split = 0
	num_vali_split = 0
	num_test_split = 0

	#모든 파일에 대해서.
	for index, i in enumerate(file):
		#get data
		data = data_read(path+i)
		data_size = len(data)


		#data shuffle
		np.random.shuffle(data)


		#data split		
		train_split = data[ : int(data_size*train_ratio) ] #소숫점 버림.
		vali_split = data[ int(data_size*train_ratio) : int(data_size * (train_ratio + vali_ratio)) ]
		test_split = data[ int(data_size * (train_ratio + vali_ratio)) : ]


		#개수 확인용 
		num_train_split += len(train_split)
		num_vali_split += len(vali_split)
		num_test_split += len(test_split)


		#make file
		with open(split_dataset_path+'train_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in train_split:
				wr.writerow(k)

		with open(split_dataset_path+'vali_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in vali_split:
				wr.writerow(k)		

		with open(split_dataset_path+'test_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in test_split:
				wr.writerow(k)

		
		print(index+1, '/', 200, ' -> ', i, 'split 완료')


	#개수 출력
	num_info = 'total : ' + str(num_train_split + num_vali_split + num_test_split) + '\n' + \
			'train_set' + '('+str(train_ratio*100)+') : ' + str(num_train_split) +'\n' + \
			'vali_set' + '('+str(vali_ratio*100)+') : ' + str(num_vali_split) +'\n' + \
			'test_set' + '('+str(test_ratio*100)+') : ' + str(num_test_split) +'\n'

	print(num_info)

	with open(split_dataset_path+'split_info.txt', 'w') as o:
		o.write(num_info)



if not os.path.isdir(split_dataset_path):
	os.mkdir(split_dataset_path)
	split_data(after_preprocess)

else:
	print("이미 폴더가 존재하므로 종료합니다.")
=======
import csv
import numpy as np
import os

after_preprocess = "./3.after_preprocess/"
split_dataset_path = "./4.split_dataset/"

train_ratio = 0.7
vali_ratio = 0.15
test_ratio = 0.15

def data_read(path):
	with open(path, 'r', newline='') as f:
		re = csv.reader(f)
		return np.array(list(re))


def split_data(path):
	file = os.listdir(path)

	num_train_split = 0
	num_vali_split = 0
	num_test_split = 0

	#모든 파일에 대해서.
	for index, i in enumerate(file):
		#get data
		data = data_read(path+i)
		data_size = len(data)


		#data shuffle
		np.random.shuffle(data)


		#data split		
		train_split = data[ : int(data_size*train_ratio) ] #소숫점 버림.
		vali_split = data[ int(data_size*train_ratio) : int(data_size * (train_ratio + vali_ratio)) ]
		test_split = data[ int(data_size * (train_ratio + vali_ratio)) : ]


		#개수 확인용 
		num_train_split += len(train_split)
		num_vali_split += len(vali_split)
		num_test_split += len(test_split)


		#make file
		with open(split_dataset_path+'train_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in train_split:
				wr.writerow(k)

		with open(split_dataset_path+'vali_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in vali_split:
				wr.writerow(k)		

		with open(split_dataset_path+'test_set.csv', 'a', newline='') as o:
			wr = csv.writer(o)
			for k in test_split:
				wr.writerow(k)

		
		print(index+1, '/', 200, ' -> ', i, 'split 완료')


	#개수 출력
	num_info = 'total : ' + str(num_train_split + num_vali_split + num_test_split) + '\n' + \
			'train_set' + '('+str(train_ratio*100)+') : ' + str(num_train_split) +'\n' + \
			'vali_set' + '('+str(vali_ratio*100)+') : ' + str(num_vali_split) +'\n' + \
			'test_set' + '('+str(test_ratio*100)+') : ' + str(num_test_split) +'\n'

	print(num_info)

	with open(split_dataset_path+'split_info.txt', 'w') as o:
		o.write(num_info)



if not os.path.isdir(split_dataset_path):
	os.mkdir(split_dataset_path)
	split_data(after_preprocess)

else:
	print("이미 폴더가 존재하므로 종료합니다.")
>>>>>>> fcdb2483f25d1b200c57673da68296fdf8b6c395
