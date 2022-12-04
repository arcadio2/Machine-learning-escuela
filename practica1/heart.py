import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set

def xyValues(file_name): 
	df = pd.read_csv(file_name, sep=',', engine='python')

	X = df.iloc[:,:13]
	y = df['target'].values 	

	return X,y,df

def validacionCruzada(X_train,y_train,k=2):
	validation_sets = []
	kf = KFold(n_splits=k)
	c=0
	for train_index, test_index in kf.split(X_train):
		print("TRAIN:", train_index, "TEST:", test_index)
		c=c+1
		#separamos el conjunto de entrenamiento a su vez en nuevo testing y nuevo entrenamiento
		#k veces
		X_train_v, X_test_v = X_train[train_index], X_train[test_index]
		y_train_v, y_test_v = y_train[train_index], y_train[test_index]
		#Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_v, y_train_v, X_test_v, y_test_v))
		print('\n PLIEGUE', c, '\n')
		print('X_train_v', *X_train_v,  '\ny_train_v', *y_train_v,'\n')
		print('X_test_v',  *X_test_v,  '\ny_test_v', *y_test_v)
	return validation_sets

def guardarArchivo(data,ruta,columnas,sep=","):
	column_names = ",".join(columnas)
	#print(column_names)
	np.savetxt(ruta, data, delimiter=sep, fmt="%d",
		header=column_names, comments="")


def generate_train_test(file_name):

	X,y,df = xyValues('heart.csv')
	column_names = df.columns.values
	column_names = column_names[:len(column_names)-1]
	#Separa corpus en conjunto de entrenamiento y prueba
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle = False)	
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	
	guardarArchivo(X_train,"train/x_e.csv",column_names)
	guardarArchivo(y_train,"train/y_e.csv",['target'])
	guardarArchivo(X_test,"test/x_p.csv",column_names)
	guardarArchivo(y_test,"test/y_p.csv",['target'])


	#Crea pliegues para la validación cruzada
	print ('----------------------')
	print('\n VALIDACION CRUZADA k=2\n') #con k=3, y bootstrap
	validation_sets = validacionCruzada(X_train,y_train,3)
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set) 
	return my_data_set,column_names

if __name__=='__main__':
	

	my_data_set,column_names=generate_train_test('heart.csv')
	print(column_names)
	#column_names = column_names[:len(column_names)-1]
	column_names = ",".join(column_names)
	#Guarda el dataset en formato csv
	
	#np.savetxt("X_test.csv", my_data_set.test_set.X_test, delimiter=",", fmt="%d",
    #       header=",".join(column_names))
	
	#np.savetxt("y_test.csv", my_data_set.test_set.y_test, delimiter=",", fmt="%d",
    #       header="y", comments="")
    
	i = 1
	for val_set in my_data_set.validation_set:
		np.savetxt("cruzada/X_train_v" + str(i) + ".csv", val_set.X_train, delimiter=",", fmt="%d",
           header=column_names, comments="")
		np.savetxt("cruzada/X_test_v" + str(i) + ".csv", val_set.X_test, delimiter=",", fmt="%d",
           header=column_names, comments="")
		np.savetxt("cruzada/y_train_v" + str(i) + ".csv", val_set.y_train, delimiter=",", fmt="%d",
           header="y", comments="")
		np.savetxt("cruzada/y_test_v" + str(i) + ".csv", val_set.y_test, delimiter=",", fmt="%d",
           header="y", comments="")
		i = i + 1
	
	#Guarda el dataset en pickle
	dataset_file = open ('dataset.pkl','wb')
	pickle.dump(my_data_set, dataset_file)
	dataset_file.close()
	
	dataset_file = open ('dataset.pkl','rb')
	my_data_set_pickle = pickle.load(dataset_file)
	print ("-----------------------------------------------")
	print (*my_data_set_pickle.test_set.X_test)

