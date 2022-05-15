import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing
import procesos
import dataframesTest
from sklearn.neural_network import MLPClassifier

#Creamos una clase Clasificador para asi tener nuestros modelos de forma individuales
class Clasificador():
    def crearModelo(self,datosTrain): #Tenemos una funcion para crear y entrenar cada modelo de acuerdo a los datos de entrenamiento dados
        scaler = StandardScaler()

        train_dataframe = procesos.getDataFrameTrain(datosTrain) #Se obtiene el dataframe de dichos datos de entrenamiento

        # Conseguimos nuestros datos de entrenamiento
        x_train = np.array(train_dataframe.feature.tolist())
        y_train = np.array(train_dataframe.class_Alofono.tolist())

        X_trainscaled = scaler.fit_transform(x_train)  # Escalamos los datos
        X_trainscaled = preprocessing.scale(X_trainscaled)  # Preprocesamos los datos

        modelo = MLPClassifier(solver='adam', alpha=1e-5, max_iter=300, hidden_layer_sizes=[40,], #Se predefine el clasificador que utilizaremos
                batch_size=14000, random_state=42).fit(X_trainscaled, y_train)
        return modelo

def predecirNodo(datoTest,modeloEntrenado,nodo): #Tenemos una funcion para predecir usando un modelo ya entrenado
    scaler = StandardScaler()

    if nodo == 'Raiz': #De acuerdo a si es un nodo Raiz
        test_dataframe = dataframesTest.getDataFrameNodoRaiz(datoTest) #Se obtiene el dataframe del archivo individual perteneciente a los datos de prueba
    elif nodo == 'Padre': # o es un nodo Parde
        test_dataframe = dataframesTest.getDataFrameNodoPadre(datoTest) #Se obtiene el dataframe del archivo individual perteneciente a los datos de prueba

    # Conseguimos nuestros datos de prueba
    x_test = np.array(test_dataframe.feature.tolist())
    y_test = np.array(test_dataframe.class_Alofono.tolist())

    X_testscaled = scaler.fit_transform(x_test) # Escalamos los datos
    X_testscaled = preprocessing.scale(X_testscaled)  # Preprocesamos los datos

    y_pred = modeloEntrenado.predict(X_testscaled) #Con el modelo entrenado tratamos de predecir el alofono

    return (accuracy_score(y_test, y_pred) * 100) #Regresamos la precision de que dicho alofono pertenezca al modelo

