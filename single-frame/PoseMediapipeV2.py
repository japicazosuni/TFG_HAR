import os
import cv2
import time
import csv
import tqdm
import numpy as np

import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("Dataset/UCF101/LongJump/v_LongJump_g01_c04.avi")
pTime = 0
count = 0
while(cap.isOpened()):    
    success, frame = cap.read()
    if frame is not None:
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
    
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("Image",frame)
        cv2.waitKey(25)
    else:
        cap.release()
        cv2.destroyAllWindows()