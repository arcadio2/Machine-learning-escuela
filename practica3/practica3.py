import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def F(w, X, y):
    #print("XD",y)
    #suma de la lista resultante
    return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
    return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)

def print_line(points, w, iteration,ax1, line_color = None, line_style = 'dotted'):
    list_x = []
    list_y = []
    for index, tuple in enumerate(points):
        x = tuple[0]
        y = x * w
        list_x.append(x)
        list_y.append(y)
    ax1.text(x,y, iteration, horizontalalignment='right')
    ax1.plot(list_x, list_y, color = line_color, linestyle= line_style)
    
    #return list_x, list_y

def regresion_lineal(X,y,i=100):
    w = 0
    alpha = 0.001
    X = list(X)
    y = list(y)
    list_error = []
    list_w = []	
    for t in range(i):
        error = F(w, X, y)
        gradient = dF(w, X, y)
        print ('gradient = {}'.format(gradient))
        #ax2.scatter(w, error)
        #ax2.text(w, error, t, horizontalalignment='right')
        list_w.append(w)
        list_error.append(error)
		
        w = w - alpha * gradient #  
        print ('iteration {}: w = {}, F(w) = {}'.format(t, w, error))
        #print_line(zip(X, y), w, t)
    return list_w,list_error

def test_train(X,Y):
    """Revolvemos los datos"""
    #columns = data.columns.values
    #data = np.array(data)
    #np.random.shuffle(data)
    #data = pd.DataFrame(data,columns=list(columns))
    pass
    #"""Obtenemos el de test y train"""
    
    #print()

def evaluacion(w,x):
    #y=wx + b 
    x = list(x)
    valores = []
    peso = w[len(w)-1] #ultimo peso
    print("XD",peso)

    for i in x: 
        y = i*peso
        valores.append(y)
        
    return valores 

def graficar(w,error,i,X_train,y_train,X_test,y_predict,y_test):

    #Graficamos la regresión con los puntos 
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Linear regression")
    ax1.set(xlabel="size", ylabel="price")
    ax1.scatter(X_train, y_train)
    for t in range(i):
        print_line(zip(X_train, y_train), w, t,ax1)
        """ list_x = []
        list_y = []
        for t in range(len(X_train)):
            pass
            x = list(X_train)[t]
            #print(x)
            y = x * w[j]
            list_x.append(x)
            list_y.append(y)
        ax1.text(x,y, j, horizontalalignment='right')
        ax1.plot(w[j], error[j], color = None, linestyle = 'dotted') """

        #list_x, list_y= print_line(zip(X, y), w, t,ax1)
        #ax1.text(X,y, t, horizontalalignment='right')
        #ax1.plot(list_x, list_y)


    #graficamos el gradiente
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Loss function")
    ax2.set(xlabel="weight", ylabel="error")

    ax2.scatter(w,error)

    #grafica la linea de error
    #print_line(zip(X, y), w, i, 'red', 'solid')
    ax2.plot(w, error, color = 'red', linestyle = 'solid')

    plt.show()
    pass


if __name__=="__main__":
    data = pd.read_csv("dataset_ejercicio.csv")
    
    X = data["size"]
    y = data["price"]
    test_train(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
    

    w,f_w = regresion_lineal(X_train,y_train,90)

    y_predict = evaluacion(w,X_test)
    print(X_test)
    print(y_predict)
    #154.21

    print(X_train)
    graficar(w,f_w,90,X_train,X_train,X_test,y_predict,y_test)


