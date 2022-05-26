import torch
from IPython.display import Image  # for displaying images
import utils # for downloading models/datasets
from roboflow import Roboflow
import sys
import os

YOLO_FILEPATH = "C:\\Users\\dylan\\Documents\\yolov5\\"
LOAD_DATASET = False
os.chdir(YOLO_FILEPATH)

device = torch.device('cuda:0')
#X = X.to(device)

print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#Load Dataset
rf = Roboflow(api_key="bu8h2C5yi0mRT1ucA7Eo")
project = rf.workspace("strawberries").project("strawberry-detect")
dataset = project.version(1).download("yolov5")

os.system("echo TRAIN TIME")
os.system("echo "+YOLO_FILEPATH+"train.py "
          "--img 416 "
          "--batch 16 "
          "--epochs 100 "
          "--data \""+dataset.location+"\\data.yaml\" "
          "--cfg "+YOLO_FILEPATH+"models\\yolov5s.yaml "
          "--weights '' "
          "--name yolov5s_results  "
          "--cache")
os.system("python3 "+YOLO_FILEPATH+"train.py "
          "--img 416 "
          "--batch 16 "
          "--epochs 100 "
          "--data \""+dataset.location+"\\data.yaml\" "
          "--cfg "+YOLO_FILEPATH+"models\\yolov5s.yaml "
          "--weights '' "
          "--name yolov5s_results  "
          "--cache")

print("Done!")
