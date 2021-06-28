import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
import cv2
import pandas as pd
import csv


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix


def plot_confusion_matrix2(cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def eliminar_imagenes(path):
    for f in os.listdir(path):
        os.remove(os.path.join(dir, f))

def generar_video(csv_path,width,height,videoName):
    try:
        # Cargar datos
        df = pd.read_csv(csv_path, header=0)
        dataset = df.values

        dim = (width, height) 
        video = cv2.VideoWriter('output/'+str(videoName)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))
        for ind in df.index:
            pathImg = "output/video/"+str(df['id_frame'][ind])+".jpg"
            imageToProcess = cv2.imread(pathImg)
            video.write(imageToProcess)

        video.release() 

    except Exception as e:
        print("gen"+e)


def split_datos(csv_path_x,csv_path_y,n_steps):
    # Target
    data_y=pd.read_csv(csv_path_y,header=None).values

    # Keypoints
    data_x=pd.read_csv(csv_path_x,header=None).values
    blocks = int(len(data_x) / n_steps)
    data_x=np.array(np.split(data_x,blocks))   

    return data_x, data_y

def split_xy(csv_path):
    # Lectura y procesamiento de los datos
    df = pd.read_csv(csv_path, header=None)
    print(df.shape)
    print(df.head())

    # Separamos keypoints de las clases
    data_key = df.iloc[0:,1:]
    print(data_key.shape)
    print(data_key.head())
    data_class = df[0]
    print(data_class.shape)
    print(data_class.head())

    X = data_key.to_numpy()
    print("--- Dimension y muestra X ---")
    print(X.shape)
    print(X[:2])
    
    # Normalizar datos
    norm = MinMaxScaler().fit(X)
    X_norm = norm.transform(X)
    print("--- Dimension y muestra X normalizada ---")
    print(X_norm.shape)
    print(X_norm[:2])

    y = data_class.values
    print("--- Dimension y muestra Y ---")
    print(y.shape)
    print(y[:2])

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)

    # Datos training
    data_trainx = open('output/data/x_train.csv', mode='w', newline="")
    data_writerx = csv.writer(data_trainx, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for keys in X_train:
        data_writerx.writerow(keys)
    
    data_trainy = open('output/data/y_train.csv', mode='w', newline="")
    data_writery = csv.writer(data_trainy, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for clase in y_train:
        data_writery.writerow([clase])

    # Datos test
    data_testx = open('output/data/x_test.csv', mode='w', newline="")
    data_writerx = csv.writer(data_testx, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for keys in X_test:
        data_writerx.writerow(keys)
    
    data_testy = open('output/data/y_test.csv', mode='w', newline="")
    data_writery = csv.writer(data_testy, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for clase in y_test:
        data_writery.writerow([clase])

def confusion_matrixUCF(datos,target):
    plot_confusion_matrix(cm=datos,normalize=False,target_names = target,title="Confusion Matrix")
