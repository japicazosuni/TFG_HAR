import sys
import os
import argparse
import pandas as pd
import numpy as np

from sys import platform

from src.procesarTracker import process_videoTracker

# Funciones fichero trex
from src.trex import extraer_keypointsTrex, knn_trex, lstm_trex, automl_trex

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
   
    ## Parametros openpose
    # Flags Openpose
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Parametros personalizados 
    params = dict()
    params["model_folder"] = "C:/TFG/openpose/models"
    params["model_pose"] = "BODY_25"

    # Variables globales
    label_names = ['avoidLeft','avoidRight','follow','load','start','stop','unload']
    csv_path = 'C:/TFG/openpose/build_windows/examples/user_code/output/process_trex.csv'
    videoToProcess = "C:/TFG/openpose/build_windows/examples/user_code/test/Test-trex_1.mp4"
    action_model = 'output/model/sentiment_analysis-trexNorm.h5'
    
    ## Valores resize img
    width = 1280
    height = 720
    dim = (width, height) 

    steps = 6

    ## Valor opcion
    opcion = 3
    
    if opcion == 0:      # Extraer caracteristicas imagenes 
        # Iniciar OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        # Puntos modelo body_25
        poseModel = op.PoseModel.BODY_25

        # Carpeta imagenes
        pathImagenes = "C:/TFG/openpose/build_windows/examples/user_code/Dataset/trex"

        extraer_keypointsTrex(pathImagenes,opWrapper,op,dim,csv_path_x,csv_path_y)

        del opWrapper
    elif opcion == 1:    # Crear modelo LSTM
        lstm_trex(csv_path,label_names,steps)
    elif opcion == 2:    # Crear modelo AutoKeras(AutoML)
        automl_trex(csv_path,label_names)
    elif opcion == 3:    # Modelo KNN
        knn_trex(csv_path,label_names)
    elif opcion == 4:    # Procesar video
        # Iniciar OpenPose
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
