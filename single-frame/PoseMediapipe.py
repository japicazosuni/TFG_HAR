import os
import cv2
import time
import mediapipe as mp
import sys


class poseDetector():
    def __init__(self, mode=False, upBody = False, smooth = True, 
        detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)
    
    def findPose(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self,img,draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return lmList

def main():
    cap = cv2.VideoCapture("HAR/test/Basketball.mp4")
    pTime = 0
    count = 0
    detector = poseDetector()
    while(cap.isOpened()):    
        success, frame = cap.read()
        if frame is not None:
            frame = detector.findPose(frame)
            lmList = detector.findPosition(frame)
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            # cv2.imwrite("frames/img_"+str(count)+".jpg",frame)
            # count+=1
            cv2.imshow("Image",frame)
            cv2.waitKey(25)
        else:
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()