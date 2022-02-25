import cv2
import numpy as np

MAX_OBJ = 100

#Read in Image
im = cv2.imread('Strawberry Test.jpg')

#Convert to Grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#test1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#test2 = np.array([5,5,5,5,5,5,5,5])
#testr = (test1 > test2)*1.0

#Threshold the image for strawbs
im_Red= (im[:,:,2] > (2.0*np.maximum(im[:,:,0],im[:,:,1])))*1.0
#cv2.threshold (im[:,:,2], im_Red, , maxval, type)


#Closing to remove noise
iter = 3
kern = cv2.getStructuringElement(cv2.MORPH_RECT, [2,2])
im_Red = cv2.dilate(im_Red,kern,iterations=iter)
im_Red = cv2.erode(im_Red,kern,iterations=iter)
im_Red = cv2.erode(im_Red,kern,iterations=iter)
im_OC = cv2.dilate(im_Red,kern,iterations=iter)

#Image Segmentation
im_OC = np.uint8(im_OC*255)
ret, markers = cv2.connectedComponents(im_OC)
#gray = cv2.applyColorMap(markers, cv2.COLORMAP_JET)

#
i=1
label = 1
im_seg = [None]*MAX_OBJ
while(1):
    seg = (markers == label)*255.0
    if sum(sum(seg)) == 0:
        break
    elif sum(sum(seg)) > 100*255:
        im_seg[i-1] = np.uint8(seg)
        i = i+1
    label = label+1

i = i-2

#Show
cv2.imshow('image',im)
cv2.imshow('Thres',im_OC)
cv2.waitKey()
while(i >= 0):
    cv2.imshow("Object: "+str(i), im_seg[i])
    i = i-1
    cv2.waitKey()

cv2.waitKey()

