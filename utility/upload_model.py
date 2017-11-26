import pickle, time, os

def write_model(model):
	path = os.path.join(os.getcwd(),'utility/model_dumps/')
	file_name = path + 'model_'+ str(time.time())

	res = pickle.dump(model, open(file_name, 'wb'))

	print('%s successfully dumped!!' % file_name)
	return res, file_name


def read_model(file_name):
	obj = pickle.load(open(file_name, 'rb'))

	print('Object loaded!!')
	return obj
