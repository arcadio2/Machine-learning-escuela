
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import operator

def sacar_promedio(data,regresion,escalado):
    filtrado = data[data["regresion"] == regresion]
    #print(filtrado)
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
    data = pd.read_csv("tabla.csv")
    print(data)

    promedios_data = promedios(data)

    min = promedios_data.loc[promedios_data["mse"] == promedios_data["mse"].min()]

    print(min)

    #buscamos donde dio el error minimo
    """Nos salio la polinomial de grado 3, creamos ese modelo"""
    df = pd.read_csv('cal_housing.csv')
    X = df.iloc[:,:-1]
    y = df["medianHouseValue"]
    columns =X.columns.values
    print(columns)

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0)

    maxi = 100000
    l = 'invscaling'
    eta = 0.0001
    polynomial_features= PolynomialFeatures(degree=3)
    
    x_poly = polynomial_features.fit_transform(X_train)

    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x_poly)
    regr = SGDRegressor(learning_rate = l, eta0 = eta, max_iter= maxi)
    regr.fit(x_scaled, y_train)

    #escalamos los datos de test
    
    x_poly_test = polynomial_features.fit_transform(X_test)
    
    x_test_scaled = scaler.transform(x_poly_test)

    y_poly_pred = regr.predict(x_test_scaled)

    mse = mean_squared_error(y_test, y_poly_pred)
    r2 = r2_score(y_test, y_poly_pred)

    print("____________________________________________________________")
    sort_axis = operator.itemgetter(0)
    
    #print (tuple(sorted_zip))
    #
    #print(pd.DataFrame(x_sorted))
    #print(pd.DataFrame(y_poly_pred))
    
    print("MSE: ",mse,"R2: ",r2)

    fig,ax = plt.subplots(2,4)
    i = 0
    j = 0
    colors = ['orange','y','b','g','brown','pink','violet','purple']
    indice = 0
    for col in columns:
        sorted_zip = sorted(zip(X_test[col],y_poly_pred), key=sort_axis)
        x_sorted, y_poly_pred = zip(*sorted_zip)

        ax[j,i].set_title(col)
        ax[j,i].set(xlabel=col, ylabel="medianHouseValue")
        ax[j,i].scatter(X_test[col],y_test,s=1.2,c=colors[indice])
        ax[j,i].plot(x_sorted,y_poly_pred,c="r")
        i+=1
        if(i==4):
            j+=1
            i=0
        indice+=1
    plt.show()

