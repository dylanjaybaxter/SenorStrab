# Script for preprocessing data
# Imports
from os import listdir
from os.path import isfile, join
import random
import pandas as pd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

# Flags
TEST_RANDOM_FILE = True
GEN_HIST = False

# Pathname
imPath = "C:\\Users\\dylan\\OneDrive - Cal Poly\\Winter 2022\\EE 428\\EE 428  Matlab Workspace\\Project 5 Workspace\\"

# Create Hists
hist_h_st = np.zeros([256], dtype="uint")
hist_s_st = np.zeros([256], dtype="uint")
hist_v_st = np.zeros([256], dtype="uint")
hist_h_bg = np.zeros([256], dtype="uint")
hist_s_bg = np.zeros([256], dtype="uint")
hist_v_bg = np.zeros([256], dtype="uint")

# Read filenames
for i in range(1, 20):
    # Read in image and mask
    im = cv.imread(imPath+"s"+str(i)+".jpg")
    mask = cv.imread(imPath+"s"+str(i)+"_roi.jpg")
    im = cv.resize(im, [mask.shape[1], mask.shape[0]])
    cv.imshow('image', im)
    cv.imshow('mask', mask)
    #cv.waitKey(0)

    # Convert to HSV
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)


    # Obtain Lists of masked and unmasked pixels
    im = np.asarray(im)
    mask = np.asarray(mask)
    px_st = im_hsv[mask == 255]
    px_bg = im_hsv[mask != 255]

    # Separate hsv
    h_st = px_st[:,0]
    s_st = px_st[:,1]
    v_st = px_st[:,2]
    h_bg = px_bg[:,0]
    s_bg = px_bg[:,1]
    v_bg = px_bg[:,2]

    # Accumulate histograms
    hist_h_st = np.add(hist_h_st, np.histogram(h_st, 256, [0, 256])[0])
    hist_s_st = np.add(hist_s_st, np.histogram(s_st, 256, [0, 256])[0])
    hist_v_st = np.add(hist_v_st, np.histogram(v_st, 256, [0, 256])[0])
    hist_h_bg = np.add(hist_h_bg, np.histogram(h_bg, 256, [0, 256])[0])
    hist_s_bg = np.add(hist_s_bg, np.histogram(s_bg, 256, [0, 256])[0])
    hist_v_bg = np.add(hist_v_bg, np.histogram(v_bg, 256, [0, 256])[0])

print("Accumulated!")
index = np.arange(0,256)
plt.subplot(2,3,1)
plt.bar(index, hist_h_st)
plt.title("Strawberry Hue")
plt.subplot(2,3,2)
plt.bar(index, hist_s_st)
plt.title("Strawberry Sat")
plt.subplot(2,3,3)
plt.bar(index, hist_v_st)
plt.title("Strawberry Val")
plt.subplot(2,3,4)
plt.bar(index, hist_h_bg)
plt.title("Background Hue")
plt.subplot(2,3,5)
plt.bar(index, hist_s_bg)
plt.title("Background Sat")
plt.subplot(2,3,6)
plt.bar(index, hist_v_bg)
plt.title("Background Val")
plt.show()

print("Done!")