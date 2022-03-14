import librosa
import numpy as np
import csv
import pandas as pd
from multiprocessing import Process,Pipe

def crear_MFCC(ruta):
    try:
        # Cargarmos el audio regresandonos la serie en el tiempo del audio y su frecuencia de muestreo
        audio, sample_rate = librosa.load(ruta)  # Cambiamos el tipo de remuestreo a kaiser_fast
        #mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # n_mfcc es el numero de MFCCs a regresar (proxima primera capa de nuestro modelo 40)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=512)  # n_fft 512

        # Calcula el promedio a traves de los axis especificados
        mfccsscaled = np.mean(mfccs.T, axis=0)  # Espectrograma de Mel de potencia logarítmica T

        # Calcula el promedio a traves de los axis especificados mfccsscaled = np.mean(mfccs.T, axis=0) #Espectrograma de Mel de potencia logarítmica T
        return mfccsscaled
    except:
        print('Error en la ruta: ',ruta)

def get_features(conn,archivo):
    mfcc_f = []
    header = True
    with open(archivo, newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            if header == True:
                header = False
                continue
            else:
                feature = crear_MFCC(row[1])
                target = int(float(row[2]))
                mfcc_f.append([feature, target])
    conn.send(mfcc_f.copy())
    conn.close()

def getDataFrame(rutaMetadata):
    procesos = []
    salida, entrada = Pipe()
    features_mfcc = []

    for n in range(8):
        proceso = Process(target=get_features, args=(entrada, str(rutaMetadata)+'/MetaData22P'+str(n+1)+'.csv',))
        procesos.append(proceso)

    for proceso in procesos:
        proceso.start()

    for proceso in procesos:
        features_mfcc += salida.recv()

    for proceso in procesos:
        proceso.join()

    featuresdf_mfcc = pd.DataFrame(features_mfcc, columns=['feature', 'class_label'])
    return featuresdf_mfcc
