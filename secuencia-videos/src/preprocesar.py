import os
import cv2
import streamlit as st
import csv

def extraerFrames(pathVideos,pathImagenes,classes):
    try:
        for clase in classes:
            path = pathVideos+""+clase
            if os.path.exists(path):
                pathOutput = pathImagenes+""+clase
                videos = os.listdir(path)
                i=0
                for video in videos:
                    # Abrir video y extraer frames
                    cap= cv2.VideoCapture(path+"/"+video)
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret == False:
                            break
                        cv2.imwrite(pathOutput+"/"+str(i)+"_"+clase+'.jpg',frame)
                        i+=1
                    cap.release()
    except Exception as e:
        st.exception(e)

# Extraer Keypoints y crear csv
def keypointsToCSV(oppwrapper,datum,csvname,pathImagenes,classes,dim,op):
    try:
        key_pointscsv = ['nose_x','nose_y','neck_x','neck_y','Rshoulder_x','Rshoulder_y','Relbow_x','Relbow_y',	'Rwrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
            'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y','LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
            'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y','LBigToe_x','LBigToe_y','LSmallToe_x','LSmallToe_y','Lheel_x','Lheel_y','RBigToe_x','RBigToe_y',
            'RSmallToe_x','RSmallToe_y','Rheel_x','Rheel_y','Background_X','Background_y','class']    

        # Creacion fichero csv
        with open('C:/TFG/openpose/build_windows/examples/user_code/output/'+csvname+'.csv', mode='w', newline="") as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)

            # Añadimos la cabecera
            data_writer.writerow(key_pointscsv)

            # Leemos frames y extraemos key points
            for clase in classes:
                path = pathImagenes+""+clase
                if os.path.exists(path):
                    imagenes = os.listdir(path)
                    for img in imagenes:
                        # datum = op.Datum()
                        imageToProcess = cv2.imread(path+"/"+img)
                        imageToProcess = cv2.resize(imageToProcess, dim, interpolation = cv2.INTER_AREA) # resize image
                        datum.cvInputData = imageToProcess
                        oppwrapper.emplaceAndPop(op.VectorDatum([datum]))
                        if datum.poseKeypoints is not None:
                            for people in datum.poseKeypoints:
                                lstKeypoints = []
                                for keypoints in people:
                                    lstKeypoints.append(keypoints[0]) #x
                                    lstKeypoints.append(keypoints[1]) #y
                                lstKeypoints.append(str(clase)) #classe
                                # Añadimos linea al fichero csv
                                data_writer.writerow(lstKeypoints)    
    except Exception as e:
        st.exception(e)
