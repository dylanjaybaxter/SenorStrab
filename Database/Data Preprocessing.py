# Script for preprocessing data
# Imports
from os import listdir
from os.path import isfile, join
import random
import pandas as pd
import cv2 as cv

#Flags
TEST_RANDOM_FILE = False
GEN_HIST = False
EXTRACT_ALL = True

#File paths
imPath = "C:\\Users\\dylan\\OneDrive - Cal Poly\\Senor Project Data\\Images\\"
anPath = "C:\\Users\\dylan\\OneDrive - Cal Poly\\Senor Project Data\\Annotations\\"
cropPath = "C:\\Users\\dylan\\OneDrive - Cal Poly\\Senor Project Data\\Crops\\"

#Get All Filenames
fn_im = [f for f in listdir(imPath) if isfile(join(imPath, f))]
fn_an = [f for f in listdir(anPath) if isfile(join(anPath, f))]
num_im = len(fn_im)
num_an = len(fn_an)
matched_list = []
unmatched_an = []
unmatched_im = []
matched = False
for im in fn_im:
    for an in fn_an:
        im_cut = im[0:-4]
        an_cut = an[0:-4]
        if im_cut == an_cut:
            matched_list.append(im_cut)
            matched = True
            break
        else:
            unmatched_im.append(an)
    if matched:
        unmatched_an.append(im)
        matched = False


random.seed(0)
if TEST_RANDOM_FILE:
    # Get a random file
    fn = random.choice(matched_list)
    # Read image and annotation
    im = cv.imread(imPath+fn+".jpg")
    an_l = pd.read_csv(anPath+fn+".csv")
    # Separate Values
    an = an_l.iloc[:,0].str.split(" ", n=4, expand=True)
    an = an.astype('int')
    # Draw Rectangles
    for row in range(0,an.shape[0]):
        [x1,y1,x2,y2] = an.iloc[row,:]
        x = x1
        y = y1
        w = abs(x2-x1)
        h = abs(y2-y1)
        cv.rectangle(im, (x,y), (x+w,y+h),(0,255,0),4)

    im = cv.resize(im, [1000, 1000])
    cv.imshow('Result with Boxes', im)
    cv.waitKey(0)

if EXTRACT_ALL:
    for fname in matched_list:
        # Read image and annotation
        im = cv.imread(imPath + fname + ".jpg")
        an_l = pd.read_csv(anPath + fname + ".csv")
        # Separate Values
        an = an_l.iloc[:, 0].str.split(" ", n=4, expand=True)
        an = an.astype('int')
        # Draw Rectangles
        for row in range(0, an.shape[0]):
            [x1, y1, x2, y2] = an.iloc[row, :]
            x = x1
            y = y1
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
            imcrop = cv.crop_img = im[y+3:y+h-3, x+3:x+w-3]
            cv.imwrite(cropPath+fname+"_"+str(row)+".jpg", imcrop)
            print("written"+fname+"_"+str(row)+".jpg")


print("Done!")