import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
from sklearn.model_selection import KFold
import pandas as pd
import random
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA

np.random.seed(0)

def regresion_lineal(X,y,max_iter=10000,learning_rate = 'constant', eta0=0.001,escalado = 0):
    #con gradiente
    #con gradiente
    if(escalado==0):
        X = X #no hacemos escalamiento
    if(escalado==1): #escalado siple}
        X = preprocessing.StandardScaler().fit_transform(X)
    else:
        X = preprocessing.RobustScaler().fit_transform(X)
    regr = SGDRegressor(learning_rate = learning_rate, eta0 = eta0, max_iter= max_iter)
    regr.fit(X,y)
    #x_poly_robust_scaler = preprocessing.RobustScaler().fit_transform(x_poly)
    y_poly_pred = regr.predict(X)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    return mse,r2,regr

def regresion_polinomial(X,y,grado=2,escalado=1,max_iter=1000,learning_rate = 'constant', eta0=0.001):
    polynomial_features= PolynomialFeatures(degree=grado)
    x_poly = polynomial_features.fit_transform(X)
    x_scaled = x_poly
    if(escalado==0):
        x_scaled = x_scaled #no hacemos escalamiento
    if(escalado==1): #escalado siple}
        x_scaled = preprocessing.StandardScaler().fit_transform(x_poly)
    else:
        x_scaled = preprocessing.RobustScaler().fit_transform(x_poly)
    regr = SGDRegressor(learning_rate = learning_rate, eta0 = eta0, max_iter= max_iter)
    regr.fit(x_scaled, y)
    y_poly_pred = regr.predict(x_scaled)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    return mse,r2,regr

def sacar_promedio(data,regresion,escalado):
    filtrado = data[data["regresion"] == regresion]
    print(filtrado)
    filtrado = filtrado[filtrado["escalamiento"]==escalado]
    mse_promedio = filtrado['mse'].mean()
    r2_promedio = filtrado['r2'].mean()

    return mse_promedio,r2_promedio

def promedios(data):
    lista_mse_promedio = []
    lista_r_promedio = []
    mse_lineal,r2_lineal = sacar_promedio(data,"lineal","sin")
    mse_poly_2_sin,r2_poly_2_sin = sacar_promedio(data,"2","sin")
    mse_poly_3_sin,r2_poly_3_sin = sacar_promedio(data,"3","sin")


    mse_lineal_simple,r2_lineal_simple = sacar_promedio(data,"lineal","simple")
    mse_poly_2_simple,r2_poly_2_simple = sacar_promedio(data,"2","simple")
    mse_poly_3_simple,r2_poly_3_simple = sacar_promedio(data,"3","simple")

    mse_lineal_robusto,r2_lineal_robusto = sacar_promedio(data,"lineal","robusto")
    mse_poly_2_robusto,r2_poly_2_robusto = sacar_promedio(data,"2","robusto")
    mse_poly_3_robusto,r2_poly_3_robusto = sacar_promedio(data,"3","robusto")


    lista_mse_promedio = [
        mse_lineal, mse_poly_2_sin, mse_poly_3_sin, mse_lineal_simple, mse_poly_2_simple, mse_poly_3_simple,
        mse_lineal_robusto,mse_poly_2_robusto,mse_poly_3_robusto
        ]
    
    lista_r_promedio = [
        r2_lineal, r2_poly_2_sin, r2_poly_3_sin, r2_lineal_simple, r2_poly_2_simple, r2_poly_3_simple, 
        r2_lineal_robusto,r2_poly_2_robusto, r2_poly_3_robusto
     ]

    data = {'mse':lista_mse_promedio,
        'r2':lista_r_promedio}
    df = pd.DataFrame(data, index =['Regresión lineal',
                                'Regresión polinomial grado 2',
                                'Regresión polinomial grado 3',
                                'Regresión lineal escalando estándar',
                                'Regresión polinomial grado 2 escalado estándar',
                                'Regresión polinomial grado 3 escalado estándar',
                                'Regresion lineal escalado robusto',
                                 'Regresión polinomial grado 2 escalado robusto',
                                'Regresión polinomial grado 3 escalado robusto'])
    print(df)
    df.to_csv("promedios.csv")
    return df


if __name__ == "__main__":
    mse_list = [] #lista de errores
    r2_list = []
    df = pd.read_csv('cal_housing.csv')
    X = df.iloc[:,:-1]
    y = df["medianHouseValue"]
    columns =X.columns.values
    print(columns)

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0)

    
    #print(X_train.values[:,0])
    
    colors = ['orange','y','b','g','brown','pink','violet','purple']
    indice = 0
    fig,ax = plt.subplots(4,2)
    i = 0
    j = 0
    for col in columns:
 
        ax[i,j].set_title(col)
        ax[i,j].set(xlabel=col, ylabel="medianHouseValue")
        ax[i,j].scatter(X_train[col],y_train,s=0.5,c=colors[indice])
        i+=1
        if(i==4):
            j+=1
            i=0
        indice+=1

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test =  y_test.values
    plt.show()
    """Creamos e conjunto de validación"""
    k = 10
    kf = KFold(n_splits=k)
    #Listas con los valores
    """ learning_rates = ['constant','optimal','invscaling','adaptive']
    eta0s = [0.01,0.005,0.001,0.0001,0.00001,0.0005,0.000001,1]
    max_iter = [9000,3000,4000,6000,7000,100000,200000] """
    #preparamos el dataset 
    maxi = 100000
    l = 'invscaling'
    eta = 0.0001
    data_regresiones = []
    data_escalamientos = []
    data_learning = []
    data_iter = []
    data_mse = []
    data_r2 = []
    data_eta = []
    for train, test in kf.split(X_train):
        X_train_v, X_test_v = X_train[train], X_train[test]
        y_train_v, y_test_v = y_train[train], y_train[test]
        #X_train_v, X_test_v = X_train.iloc[train,:], X_train.iloc[test,:]
        #y_train_v, y_test_v = y_train.iloc[train,:], y_train.iloc[test,:]
        #for l in learning_rates:
        #    for eta in eta0s:
        #        for max in max_iter:
                    #hacemos la regresion lineal nomral
        #maxi = random.choice(max_iter)
        #l = random.choice(learning_rates)
        #eta = random.choice(eta0s)
       
        """DATOS SIN ESCALAR"""
        #lineal
        mse, r2, regresion = regresion_lineal(X_train_v,y_train_v,maxi,l,eta)
        data_regresiones.append("lineal")
        data_escalamientos.append("sin")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta) 

        #plynomial grado 2
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,2,0,maxi,l,eta)
        data_regresiones.append(2)
        data_escalamientos.append("sin")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)
        
        #polynomial grado 3
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,3,0,maxi,l,eta)
        data_regresiones.append(3)
        data_escalamientos.append("sin")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)
        
        """Datos escalados"""
        #Con datos escalados

        #lineal escalado simple
        mse, r2, regresion = regresion_lineal(X_train_v,y_train_v,maxi,l,eta)
        data_regresiones.append("lineal")
        data_escalamientos.append("simple")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta) 
        #lineal escalado robusto
        mse, r2, regresion = regresion_lineal(X_train_v,y_train_v,maxi,l,eta)
        data_regresiones.append("lineal")
        data_escalamientos.append("robusto")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta) 

        #grado 2, escalado simple
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,2,1,maxi,l,eta)
        data_regresiones.append(2)
        data_escalamientos.append("simple")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)
        
        #grado 2 escalado simple
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,2,2,maxi,l,eta)
        data_regresiones.append(2)
        data_escalamientos.append("robusto")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)

        #regresion de grado 3, escalado simple
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,3,1,maxi,l,eta)
        data_regresiones.append(3)
        data_escalamientos.append("simple")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)

        #grado 3 escalado robusto
        mse, r2,regresion = regresion_polinomial(X_train_v,y_train_v,3,2,maxi,l,eta)
        data_regresiones.append(3)
        data_escalamientos.append("robusto")
        data_learning.append(l)
        data_iter.append(maxi)
        data_mse.append(mse)
        data_r2.append(r2)
        data_eta.append(eta)
        

    data = {
        "regresion":data_regresiones,
        "escalamiento":data_escalamientos,
        "learning_rate":data_learning,
        "iteraciones":data_iter,
        "eta0":eta,
        "mse":data_mse,
        "r2":data_r2
    }

    # index =[range(len(data.get('r2')))]
    df = pd.DataFrame(data)
    df.to_csv("tabla.csv")

    """Ahora encontramos los datos por separado de todos y cada uno"""
    data_promedio = promedios(df)
    #print(promedio_lineal)
    print(df)


    """Sacamos la regresion con el error minimo"""

    min = df.loc[df["mse"] == df["mse"].min()]
    print("-----------------MINIMO ERROR---------------------")
    print(min)
    print("--------------------------------------")
    min = min.values[0] #obtenemos los datos que nos dio en donde el error fue minimo
    print(min)
 

    
 
