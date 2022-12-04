import matplotlib.pyplot as plt
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def F(w, X, y):
	return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)



def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []
	for index, tuple in enumerate(points):
		x = tuple[0]
		y = x * w
		list_x.append(x)
		list_y.append(y)
	ax1.text(x,y, iteration, horizontalalignment='right')
	ax1.plot(list_x, list_y, color = line_color, linestyle= line_style)

def evaluacion(w,x):
    #y=wx + b 
	x = list(x)
	valores = []
	for i in x: 
		print(i)	
		y = i*w
		valores.append(y)
        
	return valores 



if __name__=='__main__':
	data = pd.read_csv("dataset_ejercicio.csv")
    
	X = data["size"]
	y = data["price"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)


	list_error = []
	list_w = []	
	iterations = int(500)
	#159.06/13
	#Graficamos la 
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.set_title("Regresión lineal")
	ax1.set(xlabel="size", ylabel="price")
	ax1.scatter(X, y)


	ax2 = fig.add_subplot(1, 2, 2)
	ax2.set_title("Función de perdida")
	ax2.set(xlabel="weight", ylabel="error")
	
	
	
	w= 0
	#0.000001
	alpha = 0.000001
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
	for t in range(iterations):
		error = F(w, X_train, y_train)
		gradient = dF(w, X_train, y_train)
		print ('gradient = {}'.format(gradient))
		ax2.scatter(w, error)
		ax2.text(w, error, t, horizontalalignment='right')
		list_w.append(w)
		list_error.append(error)
		
		w = w - alpha * gradient
		print ('iteration {}: w = {}, F(w) = {}'.format(t, w, error))
		print_line(zip(X_train, y_train), w, t)
			
	print("Error entrenamiento:",error)
	print_line(zip(X_train, y_train), w, t, 'red', 'solid')
	ax2.plot(list_w, list_error, color = 'red', linestyle = 'solid')
	
	plt.show()
	
	#print(sum(X_train*-y_train*2))
	
	y_predict = evaluacion(w,X_test)
	#valores de prueba
	
	y_predict = pd.DataFrame(y_predict,columns=["y_predict"],index=X_test.index,dtype=float)
	#print(y_predict)
	predictores = pd.concat([X_test,y_test,y_predict],axis=1)
	predictores.to_csv("resultado.csv")
	#predictores = pd.concat([predictores,y_predict],axis = 1)
	print(predictores)
	mse_predict = sum( (predictores["price"] - predictores["y_predict"])**2)
	print("ERROOOOOOOR")
	print(mse_predict)
	#print(y_predict)
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title("Regresión lineal contra test")
	ax1.set(xlabel="size", ylabel="price")
	

	ax1.scatter(X_test,y_predict,marker="o",c="green")
	ax1.scatter(X_test,y_test,marker="v",c="blue")
	print_line(zip(X_train, y_train), w, t, 'red', 'solid')

	plt.show()