# TFG_HAR
Este proyecto se centra en el proceso de reconocimiento y clasificación de las acciones que está realizando una persona, también conocido como Human Activity Recognition. Para realizar el reconocimiento se ha hecho uso de un algoritmo de pose estimation, el cual proporciona el esqueleto de la persona y es de ayuda para clasificar la acción que se está realizando.  Para entrenar el modelo encargado de realizar la clasificación se ha hecho uso de una red neuronal recurrente como es LSTM y también se ha hecho uso AutoML. Todos estos métodos se han puesto a prueba con datos reales para comprobar su validez.

## Contenido
A continuación, se describe el contenido de cada una de las carpetas que forman el proyecto.

* Informes: esta carpeta contiene toda la documentación generada durante el desarrollo.
* Single-frame: contiene todos los archivos utilizados durante el desarrollo y análisis, tras aplicar el método de análisis de una única imagen.
* Secuencia-videos: contiene todos los archivos de los metodos aplicados para realizar el reconocimiento de acciones a una secuencia de vídeo.
* AutoML-analisis: contiene un Jupyter Notebook el cual contiene el ejemplo de Breast Cancer y su correspondiente análisis.
* Fichero Links_interes: contiene links de distintos enlaces consultados y de interés, que he encontrado durante el desarrollo del proyecto.



## Listado de cambios
Durante el desarrollo del proyecto y entregas se han modificado los informes generando diferentes versiones.
* Informe inicial
    * PoseAnalisis-Informe InicialV1.pdf: primera versión del informe inicial.
    * PoseAnalisis-Informe InicialV2.pdf: extensión de la introducción, objetivos y metodología. También añadido apartado de planificación.
    * PoseAnalisis-InformeInicialV3.pdf: traspasada toda la información al formato final del informe y modificado apartado de planificación y añadidas algunas imagenes.

* Informe Progreso 1
    * PoseAnalisis-InformeProgreso1_V1.pdf: añadida primera versión del apartado de desarrollo en el que se han explicado los pasos realizados hasta el momento.
    * PoseAnalisis-InformeProgreso1_V2.pdf: añadidas imágenes de los resultados del análisis autoML y de reconocimiento. Extendida la introducción.

* Informe Progreso 2
    * Informe-Progreso2.pdf: revisado apartado de desarrollo respecto al análisis autoML y clasificación de poses.
    * Informe-Progreso-2.pdf: añadidas pequeñas conclusiones y introducción al apartado de reconocimiento de actividad.

* Informe final
    * Informe-final-V1.pdf: añadida parte del desarrollo de reconocimiento de poses.
    * Informe-final-V2.pdf: añadidas imágenes y toda la información del proyecto.
    * Informe-final-borrador.pdf: informe entregado como borrador tras corregir y añadir algunas imágenes.
    * Informe-final-RevisadoV2.pdf: añadidas imágenes restantes y revisión completa del texto del informe corrigiendo faltas de ortográficas y expresión escrita.

### SandBox
Imagen de los ficheros subidos al teams antes de las entregas parciales.
![SandBox Teams](/Sandbox-teams.png)

## Algoritmos
* AutoML: AutoKeras, Auto sckit-learn
* Pose Estimation: OpenPose

# Autor
* Jose Antonio Picazos Carrillo - joseantonio.picazos@e-campus.uab.cat
* Mención realizada: Computación
* Trabajo tutorizado por: Coen Antens (CVC)
* Curso 2020/21

# Referencias
*	FIFA [Online]. Disponible: https://football-technology.fifa.com/es/media-tiles/video-assistant-referee-var 
*	Kognia [Online]. Disponible: https://kogniasports.com 
*	Bnk To The Future [Online]. Disponible: https://app.bnktothefuture.com/pitches/2079/_first-v1sion-the-sports-broadcasting-revolution-with-andres-iniesta-and-serge-ibaka 
*	HomeCourt [Online]. Disponible: https://www.homecourt.ai
*	Streamlit [Online]. Disponible: https://Streamlit.io
*	AutoML [Online]. Disponible: https://www.automl.org
*	ClickUp [Online]. Disponible: https://clickup.com
*	2018 Unite the People [Online]. Disponible: https://www.simonwenkel.com/2018/12/09/Datasets-for-human-pose-estimation.html
*	2017 A History of Machine Learning and Deep Learning [Online]. Disponible: https://www.import.io/post/history-of-deep-learning/ 
*	2020 Creating a Human Pose Estimation Application with NVIDIA DeepStream [Online]. Disponible: https://developer.nvidia.com/blog/creating-a-human-pose-estimation-application-with-deepstream-sdk/  
*	MPII Human Pose Dataset [Online]. Disponible:  http://human-pose.mpi-inf.mpg.de
*	Hawk-Eye [Online]. Disponible: https://www.hawkeyeinnovations.com/ 
*	2015 Efficient and Robust Automated Machine Learning, Feurer et al., Advances in Neural Information Processing Systems 28 (NIPS 2015) [Online]. Disponible: https://proceedings.neurips.cc/paper/2015/file/11d0e6287202fced83f79975ec59a3a6-Paper.pdf 
*	Breast Cancer Wisconsin (Diagnostic) Data Set [Online]. Disponible: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 
*	2019.Haifeng Jin, Qingquan Song, and Xia Hu. "Auto-keras: An efficient neural architecture search system." Proceedings of the 25th ACM SIGKDD International Conference on Kno-wledge Discovery & Data Mining [Online]. Disponible: https://autokeras.com 
*	2012 University of Central Florida [Online]. Disponible: https://www.crcv.ucf.edu/data/UCF50.php 
*	2021 Taha Anwar. Introduction to Video Classification an Human Activity Recognition[Online]. Disponible: https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
*	2019 Adrian Rosebrock. Human Activity Recognition with OpenCV and Deep Learning [Online]. Disponi-ble:https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/ 
*	2017 The Kinetics Human Action Video Dataset [Online]. Disponible: https://arxiv.org/abs/1705.06950 
*	FitnessAlly. Disponible: https://fitnessallyapp.com/ 
 

