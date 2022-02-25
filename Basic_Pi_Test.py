import cv2

cam = cv2.VideoCapture(3)

while True:
    ret, image = cam.read()
    cv2.imshow('Imtest', image)
    key = cv2.waitKey(1)
    if key == -1:
        break
cv2.imwrite('/images/test/testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()

