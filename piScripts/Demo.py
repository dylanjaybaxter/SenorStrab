import RPi.GPIO as gp
import sys
import cv2
import numpy as np
import time

FILENAME = "test_video.avi"
FILENAME2 = "test_video_labeled.avi"
TRIGGER_PIN = 32
LED_PIN = 38

captured = False

def videoCaptureSetup2(filename1, filename2):
    video = cv2.VideoCapture(0)
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
    result2 = cv2.VideoWriter(filename2,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10,
                             size)
    return result, result2, video


# Function for Segmenting Strawberries
def strawbMask(im):
    # Threshold the image for strawbs using a normalized red
    im = cv2.GaussianBlur(im, (13, 13), cv2.BORDER_DEFAULT)
    im_Red = (im[:, :, 2] > \
        (2.0 * np.maximum(im[:, :, 0], im[:, :, 1]))) * 1.0

    # Closing to remove noise
    iter = 3
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3, 3])
    kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [4, 4])
    im_Red = cv2.erode(im_Red, kern2, iterations=iter)
    im_Red = cv2.dilate(im_Red, kern2, iterations=iter)
    im_Red = cv2.dilate(im_Red, kern, iterations=iter)
    im_OC = cv2.erode(im_Red, kern, iterations=iter)

    return im_OC


# Segment the mask
def segment(im):
    im = np.uint8(im * 255)
    ret, markers = cv2.connectedComponents(im)
    return markers


# Draw Rectangles on Image
def rectDraw(im,markers):
    # Mask Individual Segments
    im_temp = im
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
            cv2.rectangle(im_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = "DET: " + str(detected)
    coordinates = (10, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    im_temp = cv2.putText(im_temp, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    return im_temp

# Main Function
if __name__ == '__main__':

    # Setup Video Capture
    f, f2, vid = videoCaptureSetup2(FILENAME, FILENAME2)

    # Setup GPIO
    gp.setmode(gp.BOARD)
    gp.setup(LED_PIN, gp.OUT)
    gp.setup(TRIGGER_PIN, gp.IN, pull_up_down=gp.PUD_DOWN)
    gp.output(LED_PIN, gp.LOW)

    gp.output(LED_PIN, gp.HIGH)
    time.sleep(2)
    gp.output(LED_PIN, gp.LOW)

    # Global Idle
    while(True):
        # Enter Video Capture Loop When Pin is Triggered
        if gp.input(TRIGGER_PIN):
            while gp.input(TRIGGER_PIN):
                pass
            print("Capturing Video")
            cv2.destroyAllWindows()
            gp.output(LED_PIN, gp.HIGH)
            while(True):
                # Read a Frame
                ret, frame = vid.read()

                if ret == True:
                    # Write Frame to Video
                    f.write(frame)

                    # Mask the Strawbs
                    mask = strawbMask(frame)

                    # Segment
                    markers = segment(mask)

                    # Draw some rectangles
                    frame_draw = rectDraw(frame, markers)

                    # Display Video on Pi
                    # cv2.imshow('Captured Video', frame_draw)
                    f2.write(frame_draw)

                    # Exit if key is pressed or if GPIO Triggered
                    if cv2.waitKey(1) & 0xFF==ord('s') \
                            or gp.input(TRIGGER_PIN):
                        while gp.input(TRIGGER_PIN):
                            pass
                        gp.output(LED_PIN, gp.LOW)
                        cv2.destroyAllWindows()
                        print("Video Saved")
                        break
                else:
                    print("Bad Frame")
                    break
            break
        else:
            # Read a Frame
            #ret, frame = vid.read()
            # Mask the Strawbs
            #mask=strawbMask(frame)
            # Segment
            #markers = segment(mask)
            # Draw some rectangles
            #frame_draw = rectDraw(frame, markers)
            # Display Video on Pi
            #if ret == True:
                #continue
                #cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Exiting")
                break

    gp.output(LED_PIN, gp.HIGH)
    time.sleep(0.5)
    gp.output(LED_PIN, gp.LOW)
    gp.output(LED_PIN, gp.HIGH)
    time.sleep(0.5)
    gp.output(LED_PIN, gp.LOW)
    gp.output(LED_PIN, gp.HIGH)
    time.sleep(0.5)
    gp.output(LED_PIN, gp.LOW)

    cv2.destroyAllWindows()
    vid.release()
    f.release()
    f2.release()
    gp.cleanup()
    print("You did it!")
    sys.exit(0)
