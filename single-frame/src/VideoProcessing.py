import os
import sys
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

class ActivtyFrames:
    def __init__(self):
        return None

    def frames_extraction(self,video_path,image_height,image_width):
        # Empty List declared to store video frames
        frames_list = []
        
        # Reading the Video File Using the VideoCapture
        video_reader = cv2.VideoCapture(video_path)

        # Iterating through Video Frames
        while True:

            # Reading a frame from the video file 
            success, frame = video_reader.read() 

            # If Video frame was not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed Dimensions
            resized_frame = cv2.resize(frame, (image_height, image_width))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = resized_frame / 255
            
            # Appending the normalized frame into the frames list
            frames_list.append(normalized_frame)
        
        # Closing the VideoCapture object and releasing all resources. 
        video_reader.release()

        # returning the frames list 
        return frames_list


    def create_dataset(self,classes_list,dataset_directory,max_images_per_class,image_height,image_width):

        # Declaring Empty Lists to store the features and labels values.
        temp_features = [] 
        features = []
        labels = []
        
        # Iterating through all the classes mentioned in the classes list
        for class_index, class_name in enumerate(classes_list):
            print(f'Extracting Data of Class: {class_name}')
            
            # Getting the list of video files present in the specific class name directory
            files_list = os.listdir(os.path.join(dataset_directory, class_name))

            # Iterating through all the files present in the files list
            for file_name in files_list:

                # Construct the complete video path
                video_file_path = os.path.join(dataset_directory, class_name, file_name)

                # Calling the frame_extraction method for every video file path
                frames = self.frames_extraction(video_file_path,image_height,image_width)

                # Appending the frames to a temporary list.
                temp_features.extend(frames)
            
            # Adding randomly selected frames to the features list
            features.extend(random.sample(temp_features, max_images_per_class))

            # Adding Fixed number of labels to the labels list
            labels.extend([class_index] * max_images_per_class)
            
            # Emptying the temp_features list so it can be reused to store all frames of the next class.
            temp_features.clear()

        # Converting the features and labels lists to numpy arrays
        features = np.asarray(features)
        labels = np.array(labels)  

        return features, labels
    
    # Let's create a function that will construct our model
    def create_model(self,model_output_size,image_height,image_width):

        # We will use a Sequential model for model construction
        model = Sequential()

        # Defining The Model Architecture
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(model_output_size, activation = 'softmax'))

        # Printing the models summary
        model.summary()

        return model

    def plot_metric(self,model_training_history,metric_name_1, metric_name_2, plot_name):
        # Get Metric values using metric names as identifiers
        metric_value_1 = model_training_history.history[metric_name_1]
        metric_value_2 = model_training_history.history[metric_name_2]

        # Constructing a range object which will be used as time 
        epochs = range(len(metric_value_1))
        
        # Plotting the Graph
        plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
        plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
        
        # Adding title to the plot
        plt.title(str(plot_name))

        # Adding legend to the plot
        plt.legend()

    def download_youtube_videos(self,youtube_video_url, output_directory):
        # Creating a Video object which includes useful information regarding the youtube video.
        video = pafy.new(youtube_video_url)

        # Getting the best available quality object for the youtube video.
        video_best = video.getbest()

        # Constructing the Output File Path
        output_file_path = f'{output_directory}/{video.title}.mp4'

        # Downloading the youtube video at the best available quality.
        video_best.download(filepath = output_file_path, quiet = True)

        # Returning Video Title
        return video.title
    
    