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
#class data_set_final:
#	def __init__(self, validation_set):

def xyValues(file_name): 
	df = pd.read_csv(file_name, sep=',', engine='python')

	X = df.iloc[:,:22]
	y = df['RainTomorrow'].values 	

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
		#print('X_train_v', *X_train_v,  '\ny_train_v', *y_train_v,'\n')
		#print('X_test_v',  *X_test_v,  '\ny_test_v', *y_test_v)
	return validation_sets

def guardarArchivo(data,ruta,columnas,sep=","):
	column_names = ",".join(columnas)
	#print(column_names)
	np.savetxt(ruta, data, delimiter=sep, fmt="%s",
		header=column_names, comments="")


def generate_train_test():

	X,y,df = xyValues('weatherAUS.csv')
	column_names = X.columns.values
	#column_names = column_names[:len(column_names)-2]
	
	#print(y)
	print(column_names)
	X = X.values
	#Separa corpus en conjunto de entrenamiento y prueba
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)	
	
	
	guardarArchivo(X_train,"train/data_train.csv",column_names)
	guardarArchivo(y_train,"train/target_train.csv",['RainTomorrow'])
	guardarArchivo(X_test,"test/data_test.csv",column_names)
	guardarArchivo(y_test,"test/target_test.csv",['RainTomorrow'])


	#Crea pliegues para la validación cruzada
	print ('----------------------')
	print('\n VALIDACION CRUZADA k=2\n') #con k=3, y bootstrap
	validation_sets_3 = validacionCruzada(X_train,y_train,3)
	validation_sets_5 = validacionCruzada(X_train,y_train,5)
	validation_sets_10 = validacionCruzada(X_train,y_train,10)
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets_3, my_test_set) 
	my_data_set_2 = data_set(validation_sets_5, my_test_set) 

	my_data_set_3 = data_set(validation_sets_10, my_test_set) 
	my_datas = [my_data_set,my_data_set_2,my_data_set_3]


	return my_datas,column_names

def guardarConjuntos(k,my_data_set,column_names):
	i = 1
	for val_set in my_data_set.validation_set:
		np.savetxt("cruzada/data_validation_train_"+ str(k) +"_"+ str(i) + ".csv", val_set.X_train, delimiter=",", fmt="%s",
           header=column_names, comments="")
		np.savetxt("cruzada/data_test_"+ str(k) +"_"+ str(i) + ".csv", val_set.X_test, delimiter=",", fmt="%s",
           header=column_names, comments="")
		np.savetxt("cruzada/target_validation_train_"+ str(k) +"_"+ str(i) + ".csv", val_set.y_train, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		np.savetxt("cruzada/target_test_"+ str(k) +"_" +str(i) + ".csv", val_set.y_test, delimiter=",", fmt="%s",
           header="RainTomorrow", comments="")
		i = i + 1

def guardarPkl(data,name):
	#Guarda el dataset en pickle
	dataset_file = open (name,'wb')
	pickle.dump(data, dataset_file)
	dataset_file.close()
	
	dataset_file = open ( name,'rb')
	my_data_set_pickle = pickle.load(dataset_file)
	#print ("-----------------------------------------------")
	#print (*my_data_set_pickle.test_set.X_test)

if __name__=='__main__':
	

	my_data_set,column_names=generate_train_test()
	print(column_names)
	#column_names = column_names[:len(column_names)-1]
	column_names = ",".join(column_names)
	#Guarda el dataset en formato csv
	
	#np.savetxt("X_test.csv", my_data_set.test_set.X_test, delimiter=",", fmt="%d",
    #       header=",".join(column_names))
	
	#np.savetxt("y_test.csv", my_data_set.test_set.y_test, delimiter=",", fmt="%d",
    #       header="y", comments="")
    
	guardarConjuntos(3,my_data_set[0],column_names)
	guardarConjuntos(5,my_data_set[1],column_names)
	guardarConjuntos(10,my_data_set[2],column_names)
	
	guardarPkl(my_data_set[0],"dataset_3.pkl")
	guardarPkl(my_data_set[1],"dataset_5.pkl")
	guardarPkl(my_data_set[2],"dataset_5.pkl")
	

