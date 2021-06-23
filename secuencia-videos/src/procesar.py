import numpy as np
import pandas as pd
from pickle5.pickle import TRUE
from sklearn import tree
import streamlit as st
import pickle5 as pickle
import autokeras as ak
import time
import cv2
import csv
import os

from enum import Enum
from tensorflow.keras.models import load_model

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

from utiles import eliminar_imagenes,generar_video

class Actions(Enum):
    caminar = 0
    saltar = 1 
    carrera = 2 
    batida = 3 


# Cargar Modelo
def loadModelo(pathModelo):
    try:
        action_model = pickle.load(pathModelo)        
        return action_model
    except Exception as e:
        st.exception(e)

def loadModeloAutoML(pathModelo):
    try:
        action_model = load_model(pathModelo)
        return action_model
    except Exception as e:
        st.exception(e)

def loadModeloLSTM(pathModelo):
    try:
        action_model = load_model("output/model/"+pathModelo.name)
        return action_model
    except Exception as e:
        st.exception(e)

def process_imageAction(csv_path,action_modelPath,dim,autoML,lstm):
    try:
        # Cargar datos
        df = pd.read_csv(csv_path, header=0)
        st.dataframe(df.head())
        dataset = df.values
        
        # Cargar Modelo
        if autoML == True:
            action_model = loadModeloAutoML(action_modelPath)
        elif autoML == False:
            if lstm == False:
                action_model = loadModelo(action_modelPath)
            elif lstm == True:
                action_model = loadModeloLSTM(action_modelPath)

        for ind in df.index:
            pathImg = "output/img/"+str(df['id_frame'][ind])+".jpg"

            lstKeypoints = dataset[ind, 1:51].astype(float)
            
            # Reconocer accion
            if autoML == True:
                predAction = action_model.predict(np.array([lstKeypoints]))
                st.text("AutoML probabilidades->"+str(predAction))
                bestAction = np.where(predAction == np.amax(predAction))
                label_accion = "Accion: {0}".format(Actions(bestAction[0][0]).name)
            elif autoML == False:
                if lstm == False:
                    predAction = action_model.predict([lstKeypoints])
                    # Printar accion
                    label_accion = "Accion: {0}".format(Actions(predAction[0]).name)
                elif lstm == True:
                    lstKeypoints = np.array(lstKeypoints)
                    lstKeypoints = np.expand_dims(lstKeypoints, axis=0)

                    lstKeypoints = np.reshape(lstKeypoints, (lstKeypoints.shape[0], lstKeypoints.shape[1], 1))
                    predAction = action_model.predict(lstKeypoints)
                    st.text("lstm probabilidades->"+str(predAction))
                    bestAction = np.where(predAction == np.amax(predAction))
                    label_accion = "Accion: {0}".format(Actions(bestAction[0][0]).name)

            st.text(label_accion)

            # Printar imagen con accion
            imageToProcess = cv2.imread(pathImg)
            imageToProcess = cv2.resize(imageToProcess, dim, interpolation = cv2.INTER_AREA) # resize image

            cv2.putText(imageToProcess, label_accion, (50,50), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

            # Display Image
            st.image(imageToProcess, caption="Image procesada",use_column_width=True)     

        del action_model       

    except Exception as e:
        st.exception(e)


def process_imageCSV(datum,imagePath,dim,opWrapper,op):
    try:
        key_pointscsv = ['id_frame','nose_x','nose_y','neck_x','neck_y','Rshoulder_x','Rshoulder_y','Relbow_x','Relbow_y',	'Rwrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
            'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y','LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
            'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y','LBigToe_x','LBigToe_y','LSmallToe_x','LSmallToe_y','Lheel_x','Lheel_y','RBigToe_x','RBigToe_y',
            'RSmallToe_x','RSmallToe_y','Rheel_x','Rheel_y','Background_X','Background_y']   

        # Creacion fichero csv
        with open('C:/TFG/openpose/build_windows/examples/user_code/output/process_1.csv', mode='w', newline="") as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)

            # Añadimos la cabecera
            data_writer.writerow(key_pointscsv)

            id_frame = 0

            # Process Image
            imageToProcess = cv2.resize(imagePath, dim, interpolation = cv2.INTER_AREA) # resize image
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))  

            frame = datum.cvOutputData
            # label_accion = "Accion: "
            if datum.poseKeypoints is not None:
                st.text("Body keypoints: \n" + str(datum.poseKeypoints.shape))
                for people in datum.poseKeypoints:
                    lstKeypoints = []
                    lstKeypoints.append(id_frame)
                    for keypoints in people:
                        lstKeypoints.append(keypoints[0]) #x
                        lstKeypoints.append(keypoints[1]) #y

                    data_writer.writerow(lstKeypoints)

        # Guardamos Imagen 
        cv2.imwrite("output/img/"+str(id_frame)+".jpg", frame)

        # Display Image
        st.image(frame, caption="Imagen esqueleto",use_column_width=True)

        # cv2.imshow("OpenPose 1.7.0", frame )
        # cv2.waitKey(0)
    except Exception as e:
        st.exception(e)

def process_image(datum,imagePath,dim,action_model,opWrapper,op,automl,lstm):
    try:
        process_imageCSV(datum,imagePath,dim,opWrapper,op)
        del opWrapper
        del datum

        process_imageAction('C:/TFG/openpose/build_windows/examples/user_code/output/process_1.csv',action_model,dim,automl,lstm)
        del action_model

        eliminar_imagenes("output/img")
    except Exception as e:
        st.exception(e)

def process_video(pathOutputVideo,videoPath,width,height,datum,action_model,opWrapper,op,automl,lstm):
    try:
        process_videoCSV(pathOutputVideo,videoPath,width,height,datum,opWrapper,op)
        del opWrapper
        del datum

        process_videoAction('C:/TFG/openpose/build_windows/examples/user_code/output/process_2.csv',action_model,width,height,automl,lstm)
        del action_model

        generar_video('C:/TFG/openpose/build_windows/examples/user_code/output/process_2.csv',width,height,'db-normal')

        eliminar_imagenes("output/video")

    except Exception as e:
        st.exception(e)


def process_videoCSV(pathOutputVideo,videoPath,width,height,datum,opWrapper,op):
    try:
        # Process Video
        fps_count = 0
        frame_count = 0
        dim = (width, height) 

        key_pointscsv = ['id_frame','nose_x','nose_y','neck_x','neck_y','Rshoulder_x','Rshoulder_y','Relbow_x','Relbow_y',	'Rwrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
            'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y','LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
            'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y','LBigToe_x','LBigToe_y','LSmallToe_x','LSmallToe_y','Lheel_x','Lheel_y','RBigToe_x','RBigToe_y',
            'RSmallToe_x','RSmallToe_y','Rheel_x','Rheel_y','Background_X','Background_y']   

        # Creacion fichero csv
        with open('C:/TFG/openpose/build_windows/examples/user_code/output/process_2.csv', mode='w', newline="") as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)

            # Añadimos la cabecera
            data_writer.writerow(key_pointscsv)

            id_frame = 0

            cap = cv2.VideoCapture(videoPath.name)
            while (cap.isOpened()):
                # datum = op.Datum()
                hasframe, frame= cap.read()
                if hasframe== True:
                    fps_count += 1
                    frame_count += 1

                    frameToProcess = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # resize image
                    datum.cvInputData = frameToProcess
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                    frameOut = datum.cvOutputData
                    if datum.poseKeypoints is not None:
                        for people in datum.poseKeypoints:
                            lstKeypoints = []
                            lstKeypoints.append(id_frame)
                            for keypoints in people:
                                lstKeypoints.append(keypoints[0]) #x
                                lstKeypoints.append(keypoints[1]) #y

                            data_writer.writerow(lstKeypoints)
                        # Guardamos Imagen 
                        cv2.imwrite("output/video/"+str(id_frame)+".jpg", frameOut)

                        id_frame+=1
                else:
                    cap.release()
                    cv2.destroyAllWindows()
        cap.release()
    except Exception as e:
        st.exception(e)
    
def process_videoAction(csv_path,action_modelPath,width,height,autoML,lstm):
    try:
        # Cargar datos
        df = pd.read_csv(csv_path, header=0)
        st.dataframe(df.head())
        dataset = df.values

        dim = (width, height) 
        
        # Cargar Modelo
        if autoML == True:
            action_model = loadModeloAutoML(action_modelPath)
        elif autoML == False:
            if lstm == False:
                action_model = loadModelo(action_modelPath)
            elif lstm == True:
                action_model = loadModeloLSTM(action_modelPath)

        for ind in df.index:
            pathImg = "output/video/"+str(df['id_frame'][ind])+".jpg"

            lstKeypoints = dataset[ind, 1:51].astype(float)
            
            # Reconocer accion
            if autoML == True:
                predAction = action_model.predict(np.array([lstKeypoints]))
                st.text("AutoML probabilidades->"+str(predAction))
                bestAction = np.where(predAction == np.amax(predAction))
                label_accion = "Accion: {0}".format(Actions(bestAction[0][0]).name)
            elif autoML == False:
                if lstm == False:
                    predAction = action_model.predict([lstKeypoints])
                    # Printar accion
                    label_accion = "Accion: {0}".format(Actions(predAction[0]).name)
                elif lstm == True:
                    lstKeypoints = np.array(lstKeypoints)
                    lstKeypoints = np.expand_dims(lstKeypoints, axis=0)

                    lstKeypoints = np.reshape(lstKeypoints, (lstKeypoints.shape[0], lstKeypoints.shape[1], 1))
                    predAction = action_model.predict(lstKeypoints)
                    st.text("lstm probabilidades->"+str(predAction))
                    bestAction = np.where(predAction == np.amax(predAction))
                    label_accion = "Accion: {0}".format(Actions(bestAction[0][0]).name)

            # Printar imagen con accion
            imageToProcess = cv2.imread(pathImg)
            imageToProcess = cv2.resize(imageToProcess, dim, interpolation = cv2.INTER_AREA) # resize image

            cv2.putText(imageToProcess, label_accion, (50,50), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

            # Guardamos Imagen 
            cv2.imwrite(pathImg, imageToProcess)
    

        del action_model       

    except Exception as e:
        # st.exception(e)
        print(e)

def generar_video(csv_path,width,height):
    try:
        # Cargar datos
        df = pd.read_csv(csv_path, header=0)
        # st.dataframe(df.head())
        dataset = df.values

        dim = (width, height) 
        video = cv2.VideoWriter('output/db2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))
        for ind in df.index:
            pathImg = "output/video/"+str(df['id_frame'][ind])+".jpg"
            imageToProcess = cv2.imread(pathImg)
            video.write(imageToProcess)

        video.release() 

    except Exception as e:
        # st.exception(e)
        print("gen"+e)

