#imports
import cv2
import sys
from Tensorrt_Obj import TRTInference

# Define Engine Location
enginePath = "best.pt"

#Capture Video
vid_cap = cv2.VideoCapture(0)

# Setup Tensorrt Obj
Inf = TRTInference(enginePath)

while True:
    #Read in a frame
    ret, frame = vid_cap.read()

    [output, b] = Inf.infer(frame)

    #Draw Rectangles
    print(output)

    #display resulting frame
    cv2.imshow('Video',frame)

    #Exit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#
vid_cap.release()
cv2.destroyAllWindows()


