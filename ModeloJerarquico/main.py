import time
import predicciones
from sklearn import metrics
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    inicio = time.time()

    #Iniciamos las predicciones. Esta funcion llama al nodo Raiz, el cual llamara internamente
    # a los clasificadores especializados de los siguientes padres
    y_pred, y_test = predicciones.predecirAlofonos('Metadata22/Test')

    print('PRECISION MODELO JERARQUICO:', accuracy_score(y_test, y_pred) * 100) #Imprimimos la precision del modelo jerarquico

    print(f"Classification report:\n" #Ademas de sus metricas
          f"{metrics.classification_report(y_test, y_pred)}\n")

    fin = time.time()
    print('\nMinutos de ejecucion: ', (fin - inicio) / 60) #y el tiempo de ejecucion.
