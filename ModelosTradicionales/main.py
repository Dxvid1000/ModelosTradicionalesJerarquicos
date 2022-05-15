import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from procesos import getDataFrame
import time
import sys
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#Asi evitamos las alertas de usuarios
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    inicio = time.time()

    # Dividir los datos en entrenamiento y test
    test_mfcc = getDataFrame('Metadata22/Test')
    x_test = np.array(test_mfcc.feature.tolist())
    y_test = np.array(test_mfcc.class_label.tolist())

    train_mfcc = getDataFrame('Metadata22/Train')
    x_train = np.array(train_mfcc.feature.tolist())
    y_train = np.array(train_mfcc.class_label.tolist())

    #Escalamos los datos
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(x_train)
    X_testscaled = sc_X.transform(x_test)

    # Modelo 'MLP'
    """y_pred = MLPClassifier(solver='adam', alpha=2e-5, hidden_layer_sizes=(40,120),
        random_state=42,max_iter=300).fit(X_trainscaled, y_train).predict(X_testscaled)"""

    # Modelo 'SVM'
    y_pred = svm.SVC(kernel='rbf', max_iter=10000, gamma='auto', C=5.0,
                     cache_size=14000, random_state=42).fit(X_trainscaled, y_train).predict(X_testscaled)

    # Evaluar el modelo
    print('PRECISION MODELO TRADICIONAL:', accuracy_score(y_test, y_pred) * 100)

    print(f"Classification report:\n"  # Ademas de sus metricas
          f"{metrics.classification_report(y_test, y_pred)}\n")

    fin = time.time()
    print('Minutos de ejecucion: ',(fin - inicio)/60)
