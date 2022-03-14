from multiprocessing import Process,Pipe,Pool
import clasificadores
import numpy as np
import csv
import pandas as pd

def nodoPadre(rutaArchivo,modelos, numPadre):
    prob = []
    #Usamos un agregado ya que los modelos utilizados en cada nodo Padre varian haciendo que cuando un clasificador
    # especializado de una precision alta (que pertenezca a su clase) esta clase predecida concuerde con las clases preestablecidas
    if numPadre == 1: agregado = 1 #Si Padre es el 1 entonces agregamos un 1 ya que los modelos empiezan desde este indice en la lista de modelos
    elif numPadre == 2: agregado = 6 # en caso contrario agregamos un 6 ya que los modelos empiezan desde este indice en la lista de modelos
    
    for n in range(len(modelos)): #De acuerdo al numero de modelos enviados
        prob.append(clasificadores.predecirNodo(rutaArchivo, modelos[n],'Padre')) #Se consigue la probabilidad de cada uno de los modelos de ese nodo Padre

    probMayor = max(prob) #Buscamos cual es la probabilidad mayor
    for n in range(len(modelos)): #Ahora buscamos que modelo es el dueno de esa probabilidad
        if prob[n] == probMayor: #Cuando encontramos la coincidencia
            return (n+agregado) #Se regresa la clase asociada al clasificador (n) ademas del agregado

def nodoRaiz(conn, rutaMetadata,modelos,numMetadata):
    header = True
    predicciones = []

    with open(rutaMetadata, newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            if header == True:
                header = False
                continue
            else:
                rutaArchivo = str(row[1])
                probV = clasificadores.predecirNodo(rutaArchivo, modelos[0],'Raiz') #Conseguimos la probabilidad de que ese alofono sea una Vocal usando el modelo guardado en el indice 0 de la lista modelos
                probC = clasificadores.predecirNodo(rutaArchivo, modelos[1],'Raiz') #Conseguimos la probabilidad de que ese alofono sea una Consonante usando el modelo guardado en el indice 1 de la lista modelos
                if probV >= probC: #Si la probabilidad de que sea una Vocal es mayor
                    pred = nodoPadre(rutaArchivo,modelos[2:7],1) #Entonces mandamos ese alofono al nodo Padre 1 para su segunda clasificacion junto con los modelos pertenecientes a ese nodo Padre
                elif probC > probV: #En caso contrario
                    pred = nodoPadre(rutaArchivo,modelos[7:],2) #Mandamos ese alofono al nodo Padre 2 para su segunda clasificacion junto con los modelos pertenecientes a ese nodo Padre
                predicciones.append(pred) #Al final agregamos todas las clasificaciones hechas a una lista
    conn.send([predicciones,numMetadata]) #para despues regresarlas para un reordenamiento junto con el numero de metadata perteneciente ya que este servira como indice
    conn.close()

def predecirAlofonos(rutaMetadataTest): #Funcion para iniciar la prediccion de los alofonos
    modelos = []
    rutasMetadatasTrain= ['Metadata22/Train/MetadataRaiz/MetadataVocales','Metadata22/Train/MetadataRaiz/MetadataConsonantes', #Se guardan todas las rutas de los archivos para entrenamiento
        'Metadata22/Train/MetadataPadre1/MetadataA','Metadata22/Train/MetadataPadre1/MetadataE','Metadata22/Train/MetadataPadre1/MetadataI',
        'Metadata22/Train/MetadataPadre1/MetadataO','Metadata22/Train/MetadataPadre1/MetadataU','Metadata22/Train/MetadataPadre2/MetadataP','Metadata22/Train/MetadataPadre2/MetadataT',
        'Metadata22/Train/MetadataPadre2/MetadataK','Metadata22/Train/MetadataPadre2/MetadataB','Metadata22/Train/MetadataPadre2/MetadataD','Metadata22/Train/MetadataPadre2/MetadataG',
        'Metadata22/Train/MetadataPadre2/MetadataTS','Metadata22/Train/MetadataPadre2/MetadataF','Metadata22/Train/MetadataPadre2/MetadataS','Metadata22/Train/MetadataPadre2/MetadataX',
        'Metadata22/Train/MetadataPadre2/Metadata^Z','Metadata22/Train/MetadataPadre2/MetadataM','Metadata22/Train/MetadataPadre2/MetadataN', 'Metadata22/Train/MetadataPadre2/MetadataN~',
        'Metadata22/Train/MetadataPadre2/MetadataR(','Metadata22/Train/MetadataPadre2/MetadataR', 'Metadata22/Train/MetadataPadre2/MetadataL']

    # Se crean todos los modelos, los cuales se entrenan a si mismos de acuerdo a los datos dados y el clasificador elegido
    for n in range(len(rutasMetadatasTrain)):
        modelos.append(clasificadores.Clasificador().crearModelo(rutasMetadatasTrain[n])) #Se crean los modelos y se guardan el la lista modelos

    procesos = []
    salida, entrada = Pipe()
    prediccionesMetadatas = []

    for n in range(8):
        # Con los modelos creados y entrenados se envian dichos modelos para su uso en las predicciones
        proceso = Process(target=nodoRaiz, args=(entrada, rutaMetadataTest+'/MetaData22P' + str(n + 1) + '.csv', modelos,n+1))
        procesos.append(proceso)

    for proceso in procesos:
        proceso.start()

    for proceso in procesos:
        prediccionesMetadatas.append(salida.recv())

    for proceso in procesos:
        proceso.join()

    y_pred = [] #Recogemos las y predecidas

    for pred in range(1, 9):
        for p in prediccionesMetadatas: #Reacomodamos en orden de las metadatas los datos
            if p[1] == pred: #Si el indice de las predicciones es el indice que buscamos
                y_pred += p[0] #Unimos los datos del mismo a la lista de las y predecidas ordenando asi los datos

    clasesTest = []  # Conseguimos tambien los datos de prueba
    #Tenemos las clases preestablecidas
    clasesAlofonos = {'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5, 'p': 6, 't': 7, 'k': 8,
                      'b': 9, 'd': 10, 'g': 11, 'tS': 12, 'f': 13, 's': 14, 'x': 15, '^Z': 16,
                      'm': 17, 'n': 18, 'n~': 19, 'r(': 20, 'r': 21, 'l': 22}

    for n in range(8): #De acuerdo a los 8 archivos de metadata pertenecientes a los archivos de test
        header = True
        with open(rutaMetadataTest + '/MetaData22P' + str(n + 1) + '.csv', newline='') as File: #leemos cada archivo
            reader = csv.reader(File)
            for row in reader:
                if header == True:
                    header = False
                    continue
                else:
                    clasesTest.append([clasesAlofonos[row[3]]]) #y obtenemos las clases pertenecientes a cada alofono

    labels_testdf = pd.DataFrame(clasesTest, columns=['class_Alofono']) #Finalmente creamos un dataframe con las clases de los alofonos de test

    y_test = np.array(labels_testdf.class_Alofono.tolist())

    return y_pred, y_test #Por ultimo regresamos tanto los datos y predecidos como los reales
