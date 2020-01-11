from encode_faces import encodeFaces
from build_face_dataset import buildDataset
from recognize_faces_video import isUserInFrame, isLocked, recognizeFaces
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from ctypes import CDLL
import Quartz
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", action='store_true',
                    help="whether or not to display output frame to screen")
    ap.add_argument("-a", action='store_true',
                    help="whether or not to add images to dataset and retrain")
    ap.add_argument("-r", action='store_true',
                    help="whether or not to delete images and retrain from scratch")
    args = vars(ap.parse_args())

    if args['r']:
        for image in os.listdir('dataset/me'):
            os.remove('dataset/me/' + image)

    # Check if encodings file exists
    if not os.path.isfile('encodings.pickle') or any([args['a'], args['r']]):

        # If no encodings file, check if images in dataset/me directory
        if not os.listdir('dataset/me') or args['a']:

            # If no images, go through image collection script
            buildDataset()

        # Create pickle encodings file
        encodeFaces()
    print("[INFO] loading encodings...")

    # load the known faces and embeddings
    data = pickle.loads(open('encodings.pickle', "rb").read())
    imagesWithoutAuth = [1] * 91
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:

        # Only checks when state of computer is not locked
        while not isLocked():
            frame, names = recognizeFaces(args, data, vs)
            if args["d"]:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

            imagesWithoutAuth.pop()
            if len(names) == 0 or not isUserInFrame(names):
                imagesWithoutAuth.insert(0, 0)
            else:
                imagesWithoutAuth.insert(0, 1)

            if shouldLock(imagesWithoutAuth):
                print("GOING TO LOCKOUT")
                loginPF = CDLL(
                    '/System/Library/PrivateFrameworks/login.framework/Versions/Current/login')
                imagesWithoutAuth = [1] * 91
                result = loginPF.SACLockScreenImmediate()
                time.sleep(2.0)
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                vs.stop()
                cv2.destroyAllWindows()
                exit(0)

# Future improvements TODO lock out for windows users and check with os.name


def shouldLock(imagesWithoutAuth):
    if sum(imagesWithoutAuth) < 45:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
