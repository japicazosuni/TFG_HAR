import sys
import cv2
import os
import argparse
import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sys import platform
from numpy.lib.arraypad import pad
from enum import Enum

from deep_sort.iou_matching import iou_cost
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from tools.utils import poses2boxes


from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from src.procesar import process_videoTracker
from src.utiles import plot_confusion_matrix2

import autokeras as ak
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# Funcion para extraer las imagenes de los videos 
def extraer_framesUCF(path_videos,path_imagenes):
    try:
        print("##-------- Extraer Frames -------------##")

        classes = os.listdir(path_videos)
        print("--------- Clases modelo ---------------")
        print(classes)

        for clase in classes:
            path = path_videos+"/"+clase
            if os.path.exists(path):
                pathOutput = path_imagenes+"/"+clase

                if os.path.exists(pathOutput):
                    print("existe carpeta")
                else:
                    os.mkdir(pathOutput)

                videos = os.listdir(path)
                i=0
                for video in videos:
                    # Abrir video y extraer frames
                    pathV = path+"/"+video
                    cap= cv2.VideoCapture(pathV)
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret == False:
                            break
                        cv2.imwrite(pathOutput+"/"+str(i)+"_"+clase+'.jpg',frame)
                        i+=1
                    cap.release()
    except Exception as e:
        print("Frames ->"+str(e))

# Funcion para extraer los keypoints de las diferentes imagenes y almacenarlas en un csv con su respectiva clase
def extraer_keypointsUCF(path_videos,path_imagenes,opWrapper,op,dim,csv_path):
    try:
        print("##-------- Extraer Keypoints -------------##")

        # Tracker
        metric = nn_matching.NearestNeighborDistanceMetric("cosine",1,None)
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename,batch_size=1)

        # Lista de clases
        classes = os.listdir(path_videos)
        print("--------- Clases modelo ---------------")
        print(classes)

        # Listado keypoints del modelo
        key_pointscsv = ['clase','nose_x','nose_y','neck_x','neck_y','Rshoulder_x','Rshoulder_y','Relbow_x','Relbow_y',	'Rwrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
                'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y','LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
                'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y','LBigToe_x','LBigToe_y','LSmallToe_x','LSmallToe_y','Lheel_x','Lheel_y','RBigToe_x','RBigToe_y',
                'RSmallToe_x','RSmallToe_y','Rheel_x','Rheel_y','Background_X','Background_y']  

        with open(csv_path, mode='w', newline="") as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)

            data_writer.writerow(key_pointscsv)
            
            for clase in classes:
                path = path_imagenes+"/"+clase
                if os.path.exists(path):
                    imagenes = os.listdir(path)
                    for img in imagenes:
                        tracker = DeepTracker(metric,max_age=30,n_init=3) #100-20
                        datum = op.Datum()
                        imageToProcess = cv2.imread(path+"/"+img)
                        imageToProcess = cv2.resize(imageToProcess, dim, interpolation = cv2.INTER_AREA) # resize image
                        datum.cvInputData = imageToProcess
                        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                        frameOut = datum.cvOutputData

                        if datum.poseKeypoints is not None:
                            # Pintar tracker
                            keypoints = np.array(datum.poseKeypoints)
                            poses = keypoints[:,:,:2]
                            boxes = poses2boxes(poses)
                            boxes_xywh = [[x1,y1,x2-x1,y2-y1] for [x1,y1,x2,y2] in boxes]
                            features = encoder(frameOut,boxes_xywh)

                            nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
                            detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(boxes_xywh, features, poses) if nonempty(bbox)]

                            # Run non-maxima suppression.
                            boxes_det = np.array([d.tlwh for d in detections])
                            scores = np.array([d.confidence for d in detections])
                            indices = preprocessing.non_max_suppression(boxes_det, 1.0, scores)
                            detections = [detections[i] for i in indices]

                            # Llamada tracker
                            tracker.predict()
                            tracker.update(frameOut, detections)

                            for track in tracker.tracks:
                                color = None
                                if not track.is_confirmed():
                                    color = (0,0,255)
                                else:
                                    color = (255,255,255)
                                
                                if track.track_id == 1:
                                    bbox = track.to_tlbr()
                                    lstKeypoints = []
                                    lstKeypoints.append(clase)
                                    for keypoints in track.last_seen_detection.pose:
                                        lstKeypoints.append(keypoints[0]) #x
                                        lstKeypoints.append(keypoints[1]) #y
                                    data_writer.writerow(lstKeypoints)

                                    cv2.rectangle(frameOut, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
                                    cv2.putText(frameOut, "id%s - ts%s - a%s"%(track.track_id,track.time_since_update,track.age),(int(bbox[0]), int(bbox[1])-20),0, 5e-3 * 200, (0,255,0),2)
                                elif track.age > 5 and track.track_id<3:
                                    bbox = track.to_tlbr()
                                    lstKeypoints = []
                                    lstKeypoints.append(clase)
                                    for keypoints in track.last_seen_detection.pose:
                                        lstKeypoints.append(keypoints[0]) #x
                                        lstKeypoints.append(keypoints[1]) #y
                                    data_writer.writerow(lstKeypoints)

                                    cv2.rectangle(frameOut, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
                                    cv2.putText(frameOut, "id%s - ts%s - a%s"%(track.track_id,track.time_since_update,track.age),(int(bbox[0]), int(bbox[1])-20),0, 5e-3 * 200, (0,255,0),2)

                            # Guardamos Imagen 
                            if os.path.exists("output/UCF101/"+clase):
                                cv2.imwrite("output/UCF101/"+clase+"/"+str(img)+".jpg", frameOut)
                            else:
                                os.mkdir("output/UCF101/"+clase)
                                cv2.imwrite("output/UCF101/"+clase+"/"+str(img)+".jpg", frameOut)
        del datum
    except Exception as e:
        print("Keypoints ->"+str(e))

# Funcion para crear un modelo lstm en base a los keypoints extraidos de los frames y almacenados en el csv
def lstm_ucf(csv_path,label_names):
    try:
        print("##-------- Modelo LSTM -------------##")
    
        # Lectura y procesamiento de los datos
        data = pd.read_csv(csv_path)
        data = data.sample(frac=1).reset_index(drop=True)
        cols = data.columns.drop('clase')
        data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
        print(data.shape)
        print(data.head())
        y_classes = data['clase']

        dataKeypoints = data.iloc[0:,1:]
        print(dataKeypoints.head())
        print(dataKeypoints.info())

        X = dataKeypoints.to_numpy()
        print("--- Dimension y muestra X ---")
        print(X.shape)
        print(X[:5])
        
        # Normalizar datos
        norm = MinMaxScaler().fit(X)
        X_norm = norm.transform(X)
        print("--- Dimension y muestra X normalizada ---")
        print(X_norm.shape)
        print(X_norm[:5])
        
        y = pd.get_dummies(data['clase']).values
        print("--- Dimension y muestra Y ---")
        print(y.shape)
        print(y[:9])

        # Mapping de las clases
        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(y_classes)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print(class_mapping)

        # Configuracion modelo LSTM
        model = Sequential()
        model.add(Embedding(5000, 256, input_length=X_norm.shape[1]))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
        model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
        model.add(Dense(8, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # Division de datos
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)

        batch_size = 32
        epochs = 8

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        model.save('output/model/sentiment_analysis-ucf101.h5')

        # Testing model
        predictions = model.predict(X_test)

        # Matriz de confusion
        cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(predictions,axis=1))
        plot_confusion_matrix2(cm, classes=np.asarray(label_names), normalize=True,
                      title='Normalized confusion matrix') 
        plt.show()
    
    
    except Exception as e:
        print("LSTM ->"+str(e))

# Funcion para realizar el entrenamiento de un modelo KNN
def knn_ucf(csv_path,label_names):
    try:
        print("##-------- Modelo KNN -------------##")

        # Lectura y procesamiento de los datos
        data = pd.read_csv(csv_path,header=None)
        data = data.sample(frac=1).reset_index(drop=True)
        cols = data.columns.drop(0)
        data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        print("--- Dimension y muestra datos ---")
        print(data.shape)
        print(data.head())

        dataKeypoints = data.iloc[0:,1:]

        X = dataKeypoints.to_numpy()
        print("--- Dimension y muestra X ---")
        print(X.shape)
        print(X[:5])

        # Normalizar datos
        norm = MinMaxScaler().fit(X)
        X_norm = norm.transform(X)
        print("--- Dimension y muestra X normalizada ---")
        print(X_norm.shape)
        print(X_norm[:5])

        y = pd.get_dummies(data[0]).values
        [print(data[0][i], y[i]) for i in range(0,5)]
        print("--- Dimension y muestra Y ---")
        print(y.shape)
        print(y[:5])

        # Division de datos
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)
       
        # Modelo KNN con el numero de clases
        model = KNeighborsClassifier(n_neighbors=8)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        # Matriz de confusion
        plot_confusion_matrix(model, X_test, y_test, normalize='true')  
        plt.title("Matriz Confusion - k=8")
        plt.show()

        # Prueba de KNN con un rango de K
        k_range = range(1,26)
        scores = {}
        scores_list = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores[k]=metrics.accuracy_score(y_test, y_pred)
            scores_list.append(metrics.accuracy_score(y_test, y_pred))

        plt.plot(k_range,scores_list)
        plt.xlabel("Valor de K")
        plt.ylabel("Accuracy")

        plt.show()  
    except Exception as e:
        print("KNN ->"+str(e))

# Funcion para realizar un modelo mediante el algoritmo de automl de AutoKeras
def automl_ucf(csv_path,label_names):
    try:
        print("##-------- Modelo AutoML -------------##")

        # Lectura y procesamiento de los datos
        df = pd.read_csv(csv_path, header=0)
        dataset = df.values

        X = dataset[:, 1:51].astype(float)
        Y = dataset[:, 0]
        print("--- Dimension X-Y ---")
        print(X.shape)
        print(Y.shape)

        # Normalizar datos
        norm = MinMaxScaler().fit(X)
        X_norm = norm.transform(X)
        print("--- Dimension y muestra X normalizada ---")
        print(X_norm.shape)
        print(X_norm[:5])

        # Mapping de las clases
        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print("--- Mapeo clases ---")
        print(class_mapping)

        # Division de los datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_norm, encoder_Y, test_size=0.2, random_state=0)

        # Configuracion modelo AutoKeras
        clf3 = ak.StructuredDataClassifier(max_trials=1)
        clf3.fit(x=X_train, y=Y_train, epochs=50)

        # Test modelo
        y_pred_autok3 = clf3.predict(X_test)
        accuracy_autok3_df = metrics.accuracy_score(Y_test, y_pred_autok3)

        # Evaluar modelo
        print("Accuracy Evaluate: {accuracy}".format(accuracy=clf3.evaluate(X_test, Y_test)))

        # Matriz de confusion 
        cm = confusion_matrix(np.argmax(Y_train,axis=1), np.argmax(y_pred_autok3,axis=1))
        plot_confusion_matrix2(cm, classes=np.asarray(label_names), normalize=True,
                    title='Normalized confusion matrix AUTOML') 
        plt.show()

        # Obtener mejor modelo 
        best_model = clf3.tuner.get_best_model()

        # Guardar modelo
        try:
            best_model.save("output/model/automl_sentiment-ucf.h5")
        except Exception as e:
            print("Save AutoML->"+str(e))

        del clf3
    except Exception as e:
        print("AutoML Model->"+str(e))
