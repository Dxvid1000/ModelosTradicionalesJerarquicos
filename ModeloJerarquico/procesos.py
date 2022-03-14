import librosa
import numpy as np
import csv
import pandas as pd
from multiprocessing import Process,Pipe
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") #Asi evitamos las alertas de usuarios

def crear_MFCC(ruta): #Funcion para la creacion de cada MFCC
    try:
        # Cargarmos el audio regresandonos la serie en el tiempo del audio y su frecuencia de muestreo
        audio, sample_rate = librosa.load(ruta)  # Cambiamos el tipo de remuestreo a kaiser_fast
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=512)  # n_fft 512 o 2048

        # Calcula el promedio a traves de los axis especificados
        mfccsscaled = np.mean(mfccs.T, axis=0)  # Espectrograma de Mel de potencia logarítmica T

        # Calcula el promedio a traves de los axis especificados mfccsscaled = np.mean(mfccs.T, axis=0) #Espectrograma de Mel de potencia logarítmica T
        return mfccsscaled
    except:
        print('Error en la ruta: ',ruta)

def get_features(conn,archivo): #Funcion para la lectura de los archivos .csv (metadatas)
    mfcc_f = []
    header = True
    with open(archivo, newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            if header == True:
                header = False
                continue
            else:
                feature = crear_MFCC(row[1]) #Se consigue el MFCC (feature)
                alofono = int(float(row[2])) #y tambien la clase del alofono asociado a el
                mfcc_f.append([feature, alofono])
    conn.send(mfcc_f.copy())
    conn.close()

def getDataFrameTrain(rutaMetadata): #Funcion para la obtencion del dataframe de datos de acuerdo a la ruta de la metadata dada
    procesos = []
    salida, entrada = Pipe()
    features_mfcc = []

    for n in range(8):
        proceso = Process(target=get_features, args=(entrada, rutaMetadata+'/MetaData22P'+str(n+1)+'.csv',))
        procesos.append(proceso)

    for proceso in procesos:
        proceso.start()

    for proceso in procesos:
        features_mfcc += salida.recv()

    for proceso in procesos:
        proceso.join()

    featuresdf_mfcc = pd.DataFrame(features_mfcc, columns=['feature', 'class_Alofono'])
    return featuresdf_mfcc
