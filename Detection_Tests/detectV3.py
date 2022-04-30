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


    # Threshold the image for strawbs using a normalized red
    im_blur = cv2.GaussianBlur(im, (13, 13), cv2.BORDER_DEFAULT)
    im_Red_1 = (im[:,:,2] > (2.1*np.maximum(im_blur[:,:,0],im_blur[:,:,1])))*1.0

    cv2.imshow('mask 1', im_Red_1)

    # Closing to remove noise
    iter = 3
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [4, 4])
    kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3, 3])
    im_Red_2 = cv2.erode(im_Red_1, kern2, iterations=iter)
    im_Red_3 = cv2.dilate(im_Red_2, kern2, iterations=iter)
    im_Red_4 = cv2.dilate(im_Red_3, kern, iterations=iter)
    im_OC = cv2.erode(im_Red_4, kern, iterations=iter)

    # Image Segmentation
    im_OC_uint = np.uint8(im_OC * 255)
    ret, markers = cv2.connectedComponents(im_OC_uint)

    # Mask Individual Segments
    im_label = im
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
        if (w * h) > 0.1 * max_size:
            cv2.rectangle(im_label, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = "DET: "+str(detected)
    coordinates = (10, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    im_label = cv2.putText(im_label, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Blue', im[:,:,0])
    cv2.imshow('Green', im[:,:,1])
    cv2.imshow('Red', im[:,:,2])
    cv2.imshow('Red_2', im_Red_4)
    cv2.imshow('OC', im_OC*255)
    cv2.imshow('Labeled', im_label)

    key = cv2.waitKey(1)
    if key != -1:
        if DEBUG:
            print("Exiting Loop")
        break

# cv2.imwrite('/home/pi/testimage.jpg', im)
cam.release()
cv2.destroyAllWindows()