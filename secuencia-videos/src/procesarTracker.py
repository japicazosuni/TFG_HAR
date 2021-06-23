from cv2 import data
import numpy as np
import pandas as pd
from pickle5.pickle import TRUE
from sklearn import tree
import pickle5 as pickle
import autokeras as ak
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

from src.utiles import generar_video

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


class ActionsUCF(Enum):
    archery = 0
    basketball = 1 
    bowling = 2 
    golfswing = 3
    highjump = 4
    longjump = 5
    playingguitar = 6
    shotput = 7

class ActionsTrex(Enum):
    avoidLeft = 0
    avoidRight = 1 
    follow = 2 
    load = 3
    start = 4
    stop = 5
    unload = 6



def process_videoTracker(videoPath,width,height,datum,action_model,opWrapper,op):
    try:
        # process_videoTrackerCSV(videoPath,width,height,datum,opWrapper,op)
        del opWrapper
        del datum

        process_videoActionTracker('C:/TFG/openpose/build_windows/examples/user_code/output/process_2.csv',action_model,width,height)
        del action_model

        # generar_video('C:/TFG/openpose/build_windows/examples/user_code/output/process_2.csv',width,height,'db-tracker')

        # eliminar_imagenes("output/video")

    except Exception as e:
        # st.exception(e)
        print(e)


def process_videoTrackerCSV(videoPath,width,height,datum,opWrapper,op):
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
        with open('C:/TFG/openpose/build_windows/examples/user_code/output/processTracker_2.csv', mode='w', newline="") as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)

            # Añadimos la cabecera
            data_writer.writerow(key_pointscsv)

            id_frame = 0

            # Deep Tracker
            metric = nn_matching.NearestNeighborDistanceMetric("cosine",1,None)
            model_filename = 'model_data/mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename,batch_size=1)

            cap = cv2.VideoCapture(videoPath)
            while (cap.isOpened()):
                # datum = op.Datum()
                hasframe, frame= cap.read()
                if hasframe== True:
                    fps_count += 1
                    frame_count += 1
                    tracker = DeepTracker(metric,max_age=30,n_init=3) #100-20

                    frameToProcess = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # resize image
                    datum.cvInputData = frameToProcess
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
                        # Call the tracker
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
                                lstKeypoints.append(id_frame)
                                for keypoints in track.last_seen_detection.pose:
                                    lstKeypoints.append(keypoints[0]) #x
                                    lstKeypoints.append(keypoints[1]) #y
                                data_writer.writerow(lstKeypoints)

                                cv2.rectangle(frameOut, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
                                cv2.putText(frameOut, "id%s - ts%s - a%s"%(track.track_id,track.time_since_update,track.age),(int(bbox[0]), int(bbox[1])-20),0, 5e-3 * 200, (0,255,0),2)
                            elif track.age > 5 and track.track_id<3:
                                bbox = track.to_tlbr()
                                lstKeypoints = []
                                lstKeypoints.append(id_frame)
                                for keypoints in track.last_seen_detection.pose:
                                    lstKeypoints.append(keypoints[0]) #x
                                    lstKeypoints.append(keypoints[1]) #y
                                data_writer.writerow(lstKeypoints)
                                cv2.rectangle(frameOut, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
                                cv2.putText(frameOut, "id%s - ts%s - a%s"%(track.track_id,track.time_since_update,track.age),(int(bbox[0]), int(bbox[1])-20),0, 5e-3 * 200, (0,255,0),2)

                        # Guardamos Imagen 
                        cv2.imwrite("output/video/"+str(id_frame)+".jpg", frameOut)

                        id_frame+=1
                else:
                    cap.release()
                    cv2.destroyAllWindows()
        cap.release()
    except Exception as e:
        # st.exception(e)
        print("vt "+str(e))
    
def process_videoActionTracker(csv_path,action_modelPath,width,height):
    try:
        cols = ['nose_x','nose_y','neck_x','neck_y','Rshoulder_x','Rshoulder_y','Relbow_x','Relbow_y',	'Rwrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
                    'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y','LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
                    'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y','LBigToe_x','LBigToe_y','LSmallToe_x','LSmallToe_y','Lheel_x','Lheel_y','RBigToe_x','RBigToe_y',
                    'RSmallToe_x','RSmallToe_y','Rheel_x','Rheel_y','Background_X','Background_y'] 
        # Lectura y procesamiento de los datos
        df = pd.read_csv(csv_path, header=0)
        dataset = df.values
        print(df.head()) 
        print(dataset[:5,1:51])

        # Normalizar MinMax
        # norm = MinMaxScaler().fit(dataset[0:,1:51])
        # dataset[0:,1:51] = norm.transform(dataset[0:,1:51])
        # print("Datos normalizados MinMax")
        # print(dataset.shape)
        # print(dataset)

        # Normalizar
        normalized_arr = preprocessing.normalize(dataset[0:,1:51])
        print("Datos normalizados")
        print(normalized_arr)
        dataset = normalized_arr

        dim = (width, height) 
        
        # Cargar modelo
        action_model = load_model(action_modelPath)

        frames_lstkeypoints = []
        most_common_action = ''
        frame = 0
        for ind in df.index:
            pathImg = "output/video/"+str(df['id_frame'][ind])+".jpg"
            if frame==10: 
                # Reconocer accion
                predAction = action_model.predict(np.array(frames_lstkeypoints))
                # Obtener el mejor de cada prediccion y escoger el que mas repetido salga?
                lst_best = []
                bestAction = ''
                for best in predAction:
                    bestAction = np.where(best == np.amax(best))
                    lst_best.append((bestAction[0][0]))
                    most_common_action= max(lst_best, key = lst_best.count)
                
                print("Common-->"+str(most_common_action))
                print("Action common --> "+ str(ActionsTrex(most_common_action).name))
                # label_accion = "Accion: {0}".format(ActionsUCF(most_common_action).name)
                label_accion = "Accion: {0}".format(ActionsTrex(most_common_action).name)

                # Reiniciamos variables
                frame = 0  
                frames_lstkeypoints = []

                # Añadir keypoint
                lstKeypoints = dataset[ind, 1:51].astype(float)
                frames_lstkeypoints.append(lstKeypoints)
                frame+=1        
            else:
                lstKeypoints = dataset[ind, 1:51].astype(float)
                frames_lstkeypoints.append(lstKeypoints)
                if most_common_action != '':
                    # label_accion = "Accion: {0}".format(ActionsUCF(most_common_action).name)
                    label_accion = "Accion: {0}".format(ActionsTrex(most_common_action).name)
                else:
                    label_accion = "Accion: "
                frame+=1    
                        

            # Printar imagen con accion
            if os.path.isfile(pathImg):
                imageToProcess = cv2.imread(pathImg)
                imageToProcess = cv2.resize(imageToProcess, dim, interpolation = cv2.INTER_AREA)

                cv2.putText(imageToProcess, label_accion, (50,50), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

                # Guardamos Imagen 
                cv2.imwrite(pathImg, imageToProcess)    

        del action_model       

    except Exception as e:
        # st.exception(e)
        print("va-->"+str(e))