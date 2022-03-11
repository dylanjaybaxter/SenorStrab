import RPi.GPIO as gp
import sys
import cv2

FILENAME = "test_video.avi"
TRIGGER_PIN = 40
LED_PIN = 38

def videoCaptureSetup(filename):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Capture Failed")
        return None
    w = int(video.get(3))
    h = int(video.get(4))
    size = (w,h)
    result = cv2.VideoWriter(filename,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10,
                             size)
    return result, video

# Main Function
if __name__ == '__main__':

    #Setup Video Capture
    f, vid = videoCaptureSetup(FILENAME)

    #Setup GPIO
    gp.setup(LED_PIN, gp.OUT)
    gp.setup(TRIGGER_PIN, gp.IN, pull_up_down=gp.gp.PUD_DOWN)
    gp.output(LED_PIN, gp.LOW)

    #Global Idle
    while(True):
        #Enter Video Capture Loop When Pin is Triggered
        if gp.input(TRIGGER_PIN):
            print("Capturing Video")
            cv2.destroyAllWindows()
            gp.ouput(LED_PIN, gp.HIGH)
            while(True):
                #Read a Frame
                ret, frame = vid.read()

                if ret == True:
                    #Write Frame to Video
                    f.write(frame)

                    #Display Video on Pi
                    cv2.imshow('Captured Video', frame)

                    #Exit if key is pressed or if GPIO Triggered
                    if cv2.waitKey(1) & 0xFF==ord('s') \
                            or gp.input(TRIGGER_PIN):
                        vid.release()
                        f.release()
                        gp.output(LED_PIN, gp.LOW)
                        cv2.destroyAllWindows()
                        print("Video Saved")
                        break
                else:
                    print("Bad Frame")
                    break

        # Write Frame to Video
        f.write(frame)

        # Display Video on Pi
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Exiting")
            break

    cv2.destroyAllWindows()
    gp.cleanup()
    print("You did it!")
    sys.exit(0)
