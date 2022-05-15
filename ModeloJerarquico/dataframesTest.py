import librosa
import numpy as np
import pandas as pd

def crear_MFCC(ruta):
    try:
        # Cargarmos el audio regresandonos la serie en el tiempo del audio y su frecuencia de muestreo
        audio, sample_rate = librosa.load(ruta)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=512)

        # Calcula el promedio a traves de los axis especificados
        mfccsscaled = np.mean(mfccs.T, axis=0)  # Espectrograma de Mel de potencia logarítmica T

        # Calcula el promedio a traves de los axis especificados mfccsscaled = np.mean(mfccs.T, axis=0) #Espectrograma de Mel de potencia logarítmica T
        return mfccsscaled
    except:
        print('Error en la ruta: ',ruta)

def getDataFrameNodoRaiz(ruta): #Funcion para conseguir el dataframe del archivo dado en el nodo Raiz
    features_mfcc = []
    # Tenemos las clases predefinidas para las vocales [0] y consonantes [1]
    clasesAlofonos = {'p': 1, 't': 1, 'k': 1, 'b': 1, 'd': 1, 'g': 1, 'tS': 1,
                      'f': 1, 's': 1, 'x': 1, '^Z': 1, 'm': 1, 'n': 1, 'n~': 1, 'r(': 1,
                      'r': 1, 'l': 1, 'i': 0, 'u': 0, 'e': 0, 'o': 0, 'a': 0}
    feature = crear_MFCC(ruta) #Se crea el MFCC de dicho archivo (feature)
    alofono = int(clasesAlofonos[(ruta.split('/')[2])]) #tambien se consigue la clase del alofono
    features_mfcc.append([feature, alofono]) #Se agrega el feature y alofono a la lista features_mfcc
    featuresdf_mfcc = pd.DataFrame(features_mfcc, columns=['feature', 'class_Alofono']) #para despues crear el dataframe
    return featuresdf_mfcc

def getDataFrameNodoPadre(ruta): #Funcion para conseguir el dataframe del archivo dado en el nodo Padre
    features_mfcc = []
    # Tenemos las clases predefinidas para los alofonos en los nodos Padre
    clasesAlofonos = {'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5, 'p': 6, 't': 7, 'k': 8,
                      'b': 9, 'd': 10, 'g': 11, 'tS': 12, 'f': 13, 's': 14, 'x': 15, '^Z': 16,
                      'm': 17, 'n': 18, 'n~': 19, 'r(': 20, 'r': 21, 'l': 22}
    feature = crear_MFCC(ruta) #Se crea el MFCC de dicho archivo (feature)
    alofono = int(clasesAlofonos[(ruta.split('/')[2])]) #tambien se consigue la clase del alofono
    features_mfcc.append([feature, alofono]) #Se agrega el feature y alofono a la lista features_mfcc
    featuresdf_mfcc = pd.DataFrame(features_mfcc, columns=['feature', 'class_Alofono']) #para despues crear el dataframe
    return featuresdf_mfcc
