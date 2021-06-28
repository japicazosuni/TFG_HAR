import streamlit as st
import numpy as np
import pandas as pd
import csv
import os
import sys
import cv2
import time
import tempfile
import matplotlib.pyplot as plt
import sklearn.metrics as metrics



from sys import platform
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import autokeras as ak


from src.training import autokeras_model,randomForestModel,mlpModel,gpcModel,LSTMModel
from src.procesar import process_image,process_video
from src.preprocesar import extraerFrames, keypointsToCSV

st.set_option('deprecation.showPyplotGlobalUse', False)

CLASSIFIERS = {
    "Random Forest": 1,
    "AutoML": 2,
    "MLP": 3,
    "Gaussian Process": 4,
    "LSTM": 5
}

ACCIONES = {
    "---" : 0,
    "Extraer Caracteristicas": 1,
    "Entrenar Modelo": 2,
    "Probar Modelo": 3,
}


# Variables globales
classes = ['caminar','saltar','carrera','batida']
pathVideos = "C:/TFG/openpose/build_windows/examples/user_code/Dataset/Videos/"
pathImagenes = "C:/TFG/openpose/build_windows/examples/user_code/Dataset/Imagenes/"
pathOutputVideo = "C:/TFG/openpose/build_windows/examples/user_code/output/test_video.mp4"


# Cargar OpenPose
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
        st.error('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
except Exception as e:
    st.exception(e)

# Side Bar
st.sidebar.header('Pose Analisis')
st.sidebar.text("Explicación proyecto")

# Side Bar - Variables globales
st.sidebar.subheader("Parametros generales")
st.sidebar.text("Tamaños redimension imagen")
width = st.sidebar.number_input('Width:',1280)
height = st.sidebar.number_input('Height:',720)
dim = (width, height) 

st.sidebar.subheader("Parametros OpenPose")
body_model = st.sidebar.selectbox("Modelo Body?",["BODY_25"])
params = dict()
params["model_folder"] = "C:/TFG/openpose/models"
params["model_pose"] = body_model

# Side Bar - Seleccionar Accion
act = st.sidebar.selectbox("Accion:",list(ACCIONES.keys()))
accion = ACCIONES[act]

if accion == 1:
    # Side Bar - Extraer keypoints y crear csv
    st.subheader("Extraer caracteristicas imagenes")
    st.text("Esta acción extrae las caracteristicas de las imagenes (keypoints) y las almacena en el csv")
    if st.button("Extraer frames videos"):
        with st.spinner('Extrayendo frames...'):
            extraerFrames(pathVideos,pathImagenes,['carrera','batida'])

    ckimagenes = st.checkbox('Imagenes generadas', True)
    csvname = st.text_input('Nombre csv:','data_2')
    if csvname is not None:
        if st.button("Iniciar proceso"):
            st.text("Extraer keypoints")
            
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            datum = op.Datum()
            with st.spinner('Extrayendo keypoints...'):
                keypointsToCSV(opWrapper,datum,csvname,pathImagenes,classes,dim,op)

            del opWrapper
            del datum
elif accion == 2:
    # Side Bar - Entrenar Modelo
    st.header("Entrenar Model")
    st.subheader("Parametros")
    a = st.selectbox("Clasificador?",list(CLASSIFIERS.keys()))
    filecsv = st.file_uploader("Datos csv")
    namemodel = st.text_input('Nombre modelo:','action_recognition')
    test_size = st.number_input('Test size:',0.3)

    if CLASSIFIERS[a] == 1:
        if filecsv is not None:
            if st.button("Iniciar proceso"):
                st.text("Generando Modelo con Random Forest")
                with st.spinner('Generando modelo...'):
                    randomForestModel(filecsv,namemodel,test_size)
    elif CLASSIFIERS[a] == 2:
        if filecsv is not None:
            if st.button("Iniciar proceso"):
                with st.spinner('Generando modelo...'):
                    st.text("Generando Modelo con AutoSklearn")
                    autokeras_model(filecsv,namemodel,test_size)
    elif CLASSIFIERS[a] == 3:
        if filecsv is not None:
            if st.button("Iniciar proceso"):
                with st.spinner('Generando modelo...'):
                    st.text("Generando Modelo con MLP")
                    mlpModel(filecsv,namemodel,test_size)
    elif CLASSIFIERS[a] == 4:
        if filecsv is not None:
            if st.button("Iniciar proceso"):
                with st.spinner('Generando modelo...'):
                    st.text("Generando Modelo con GPC")
                    gpcModel(filecsv,namemodel,test_size)
    elif CLASSIFIERS[a] == 5:
        if filecsv is not None:
            if st.button("Iniciar proceso"):
                with st.spinner('Generando modelo...'):
                    st.text("Generando Modelo con LSTM")
                    LSTMModel(filecsv,namemodel,test_size)
elif accion == 3:
    st.header("Probar modelo")
    st.subheader("Parametros")

    # Side Bar - Cargar Modelo
    st.text("Cargar Modelo")
    automl = st.checkbox("Es un modelo de AutoML?",False)
    lstm = st.checkbox("Es un modelo LSTM?",False)
    if automl == True:
        st.text("Indicar ruta a la carpeta del modelo")
        filemodelo = st.text_input("Ruta Modelo","C:/TFG/openpose/build_windows/examples/user_code/output/model/action_recognitionauto")
    elif automl == False:
        filemodelo = st.file_uploader("Modelo")    

    # action_model = None
    # if filemodelo is not None:
    #     st.text("Cargando Modelo")
    #     if automl == False:
    #         st.text("Fichero modelo -> "+str(filemodelo.name))
    #         action_model = loadModelo(filemodelo)
    #     elif automl == True:
    #         action_model = loadModeloAutoML(filemodelo)
    # Side Bar - Validar modelo
    if filemodelo is not None:
        validar = st.selectbox("Validar?",["Video","Imagen"])
        if validar == "Video":
            st.text("Opciones video")
            video = st.file_uploader("Video")
            if video is not None:
                if st.button("Iniciar proceso"):
                    # Starting OpenPose
                    opWrapper = op.WrapperPython()
                    opWrapper.configure(params)
                    opWrapper.start()
                    datum = op.Datum()

                    tvideo = tempfile.NamedTemporaryFile(delete=False)
                    tvideo.write(video.read())
                    with st.spinner('Procesando video...'):
                        process_video(pathOutputVideo,tvideo,width,height,datum,filemodelo,opWrapper,op,automl,lstm)

                        video_file = open('output/db1.mp4', 'rb')
                        video_bytes = video_file.read()
                        st.video(video_file)
                    
                    del opWrapper
                    del datum

        elif validar == "Imagen":
            st.text("Opciones imagen")
            img = st.file_uploader("Imagen")
            if img is not None:
                if st.button("Iniciar proceso"):
                    # Starting OpenPose
                    opWrapper = op.WrapperPython()
                    opWrapper.configure(params)
                    opWrapper.start()
                    datum = op.Datum()

                    imageprocess = Image.open(img)
                    imageprocess = np.array(imageprocess)
                    st.image(imageprocess, caption="Image antes de procesar",use_column_width=True)
                    with st.spinner('Procesando imagen...'):
                        process_image(datum,imageprocess,dim,filemodelo,opWrapper,op,automl,lstm)

                    del opWrapper
                    del datum
                


    

    



