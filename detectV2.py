"""
Strawberry Detection Test Script for Raspberry Pi
Author: Dylan Baxter
Date Modified: 4/9/22
Description: This script is intended to read a video file and perform strawberry detection on it
"""
import cv2
import numpy as np

#Define Parameters
DEBUG = 1
FIRST = 1
MAX_OBJ = 100

#Open Camera Module
cam = cv2.VideoCapture("Strawb_test_vid_1.mp4")
fps = cam.get(cv2.CAP_PROP_FPS)
print("Starting Video Capture")

#Loop Image Capture
while True:
    # Read in Image
    ret, im = cam.read()
    if ret == False:
        break
    im = cv2.resize(im, [300, 300])

    if DEBUG & FIRST:
        print("Read in first Image")
        FIRST = 0

    # Convert image to hsv
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


    # Threshold the image for strawbs
    hue_hsv = im_hsv[:,:,0]
    sat_hsv = im_hsv[:, :, 1]
    val_hsv = im_hsv[:, :, 2]
    im_hsv_blur = cv2.GaussianBlur(im_hsv, (13,13), cv2.BORDER_DEFAULT)
    #sat_hsv_blur = cv2.GaussianBlur(sat_hsv, (5, 5), 0)
    #val_hsv_blur = cv2.GaussianBlur(val_hsv, (5, 5), 0)
    red_low_1 = np.array([150, 100, 100])
    red_high_1 = np.array([179, 255, 255])
    red_low_2 = np.array([0, 100, 100])
    red_high_2 = np.array([50, 255, 255])
    mask1 = cv2.inRange(im_hsv_blur, red_low_1, red_high_1)
    mask2 = cv2.inRange(im_hsv_blur, red_low_2, red_high_2)
    im_Red_1 = cv2.bitwise_or(mask1, mask2)

    cv2.imshow('mask 1', mask1)
    cv2.imshow('mask 2', mask2)

    '''im_Red = (((hue_hsv < 0.02*255) | (hue_hsv < 0.94*255)) &
              ((sat_hsv > 0.4*255) & (sat_hsv < 0.8*255)) &
              ((val_hsv > 75) & (val_hsv < 210))) * 1.0'''

    # Closing to remove noise
    iter = 3
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, [5, 5])
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, [2, 2])
    im_Red = cv2.erode(im_Red_1, kern2, iterations=iter)
    im_Red = cv2.dilate(im_Red, kern2, iterations=iter)
    im_Red = cv2.dilate(im_Red, kern, iterations=iter)
    im_OC = cv2.erode(im_Red, kern, iterations=iter)

    # Image Segmentation
    im_OC = np.uint8(im_OC * 255)
    ret, markers = cv2.connectedComponents(im_OC)

    # Mask Individual Segments
    detected = 0
    label = 1
    im_seg = [None] * 50
    max_size = 0
    while (1):
        seg = (markers == label) * 255.0
        if sum(sum(seg)) == 0:
            break
        elif sum(sum(seg)) > 100 * 255:
            im_seg[detected] = np.uint8(seg)
            detected = detected + 1
        label = label + 1

    # Find Max Rectangle Size
    for i in range(0, detected):
        x, y, w, h = cv2.boundingRect(im_seg[i])
        if (w * h) > max_size:
            max_size = (w * h)

    # Show Results if 70% max size or larger
    for i in range(0, detected):
        x, y, w, h = cv2.boundingRect(im_seg[i])
        if (w * h) > 0.7 * max_size:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detecting', im_hsv)
    cv2.imshow('hue', hue_hsv)
    cv2.imshow('sat', sat_hsv)
    cv2.imshow('val', val_hsv)
    cv2.imshow('red1', im_Red_1)
    cv2.imshow('OC', im_OC*255)
    cv2.imshow('Red Channel', im[:, :, 1])


    key = cv2.waitKey(1)
    if key != -1:
        if DEBUG:
            print("Exiting Loop")
        break

#cv2.imwrite('/home/pi/testimage.jpg', im)
cam.release()
cv2.destroyAllWindows()