import cv2
import numpy as np

#Define Parameters
DEBUG = 1
FIRST = 1
MAX_OBJ = 100

#Open Camera Module
cam = cv2.VideoCapture(0)
print("Starting Video Capture")

#Loop Image Capture
while True:
    # Read in Image
    ret, im = cam.read()
    if DEBUG & FIRST:
        print("Read in first Image")
        FIRST = 0

    # Convert to Grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Threshold the image for strawbs
    im_Red = (im[:, :, 2] > (2.0 * np.maximum(im[:, :, 0], im[:, :, 1]))) * 1.0

    # Closing to remove noise
    iter = 3
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, [2, 2])
    im_Red = cv2.dilate(im_Red, kern, iterations=iter)
    im_Red = cv2.erode(im_Red, kern, iterations=iter)
    im_Red = cv2.erode(im_Red, kern, iterations=iter)
    im_OC = cv2.dilate(im_Red, kern, iterations=iter)

    # Image Segmentation
    im_OC = np.uint8(im_OC * 255)
    ret, markers = cv2.connectedComponents(im_OC)

    # Mask Individual Segments
    i = 1
    label = 1
    im_seg = [None] * MAX_OBJ
    while (1):
        seg = (markers == label) * 255.0
        if sum(sum(seg)) == 0:
            break
        elif sum(sum(seg)) > 100 * 255:
            im_seg[i - 1] = np.uint8(seg)
            i = i + 1
        label = label + 1
    i = i - 2

    # Show Results
    while (i >= 0):
        x,y,w,h = cv2.boundingRect(im_seg[i])
        cv2.rectangle(im, (x,y), (x+w,y+h),(0,255,0),2)
        i = i - 1
    cv2.imshow('Detecting', im)

    key = cv2.waitKey(1)
    if key != -1:
        if DEBUG:
            print("Exiting Loop")
        break

cv2.imwrite('/home/pi/testimage.jpg', im)
cam.release()
cv2.destroyAllWindows()
