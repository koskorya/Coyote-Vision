from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os


def buildDataset():
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("[INFO] starting video stream for image collection training...")
    # Change src=1 if using a usb camera
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    total = len(os.listdir('dataset/me'))
    print("[INFO] Press `k` to capture an image, press `q` when you are done adding images to train")
    print("[INFO] Make sure their is a green box around your face when you capture the image")

    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        orig = frame.copy()
        frame = imutils.resize(frame, width=750)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Coyote-Vision Training", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `k` key was pressed, write the *original* frame to disk
        if key == ord("k"):
            p = os.path.sep.join(
                ['dataset/me', "{}.png".format(str(total).zfill(5))])
            cv2.imwrite(p, orig)
            total += 1
            print(f"total images:{total} press `q` to begin training on these images")

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            vs.stop()
            cv2.destroyAllWindows()
            time.sleep(2.0)
            break

    print("[INFO] {} face images stored".format(total))
    return
