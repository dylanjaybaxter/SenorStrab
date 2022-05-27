#imports
import cv2
import torch

def videoCaptureSetup(filename1, capture):
    video = cv2.VideoCapture(capture)
    if not video.isOpened():
        print("Capture Failed")
        video = None
    w = int(video.get(3))
    h = int(video.get(4))
    size = (w,h)
    result = cv2.VideoWriter(filename1,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10,
                             size)
    return result, video

#Load Cascade Classifier
# Set to default webcam, Can also provide video filename
# if ffmpeg is installed, as opencv cannot decode on its own
# Must be compiled into opencv directly
model_path = "yolo5m_P_v6.pt"
#sourcePath = "test_video.avi"
sourcePath = 0 # Webcam
#sourcePath = "C:\\Users\\dylan\\Documents\\yolov5\\Strawberry-Detect-1\\valid\\images"

#Capture Video
f,vid_cap = videoCaptureSetup("labeled_YOLO_Output.avi", sourcePath)

# Setup Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.cuda()
model.iou = 0.5
model.max_det = 100

while True:
    #Read in a frame
    ret, frame = vid_cap.read()

    #Convert to grayscale
    #frame = cv2.resize(frame, (416, 416))
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results.print()
    boxes = results.pandas().xyxy[0]
    print(boxes)
    #Draw Rectangles
    for index,row in boxes.iterrows():
        x1 = int(row['xmin'])
        x2 = int(row['xmax'])
        y1 = int(row['ymin'])
        y2 = int(row['ymax'])
        print(x1, x2, y1, y2)
        cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,255), 2)
        cv2.putText(frame, row['name']+": "+str(round(row['confidence']*100, 1))+"%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)

    #display resulting frame
    cv2.imshow('Video',frame)
    f.write(frame)

    #Exit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f.release()
vid_cap.release()
cv2.destroyAllWindows()


