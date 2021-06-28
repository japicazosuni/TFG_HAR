import sys
import os
import argparse

from sys import platform
from enum import Enum

from src.procesarTracker import process_videoTracker

# Funciones fichero ucf
from src.ucf101 import extraer_framesUCF, extraer_keypointsUCF, knn_ucf, lstm_ucf, automl_ucf

class Actions(Enum):
    archery = 0
    basketball = 1 
    bowling = 2 
    golfswing = 3
    highjump = 4
    longjump = 5
    playingguitar = 6
    shotput = 7

try:

    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Parametros openpose
    ## Flags Openpose
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    ## Parametros personalizados
    params = dict()
    params["model_folder"] = "C:/TFG/openpose/models"
    params["model_pose"] = "BODY_25"

    # Variables globales
    pathVideos = "C:/TFG/openpose/build_windows/examples/user_code/Dataset/UCF101"
    pathImagenes = "C:/TFG/openpose/build_windows/examples/user_code/Dataset/UCF101_img"
    videoToProcess = "C:/TFG/openpose/build_windows/examples/user_code/test/LongJump-test_trim.mp4"
    csv_path = 'C:/TFG/openpose/build_windows/examples/user_code/output/process_ucf101.csv'
    action_model = 'output/model/sentiment_analysis-ucf101.h5'
    label_names = ['archery','basketball','bowling','golfswing','highjump','longjump','playingguitar','shotput']

    ## Valores resize img
    width = 1280
    height = 720
    dim = (width, height) 

    ## Valor opcion
    opcion = 5

    if opcion == 0:     # Extraer frames videos
        extraer_framesUCF(pathVideos,pathImagenes)
    elif opcion == 1:   # Extraer keypoints frames videos
        # Iniciar OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        # Puntos modelo body_25
        poseModel = op.PoseModel.BODY_25

        extraer_keypointsUCF(pathVideos,pathImagenes,opWrapper,op,dim,csv_path)
        
        del opWrapper
    elif opcion == 2:   # Crear Modelo LSTM
        lstm_ucf(csv_path,label_names)      
    elif opcion == 3:   # Crear Modelo KNN
        knn_ucf(csv_path,label_names)
    elif opcion == 4:   # Crear modelo AutoKeras(AutoML)
        automl_ucf(csv_path,label_names)
    elif opcion == 5:   # Procesar video
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        datum = op.Datum()       

        process_videoTracker(videoToProcess,width,height,datum,action_model,opWrapper,op)
        
        del datum 
        del opWrapper
except Exception as e:
    print(e)
    sys.exit(-1)
