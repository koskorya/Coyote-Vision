from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


def encodeFaces():
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images('dataset'))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # cnn is more accurate but about 7 seconds per image
        boxes = face_recognition.face_locations(rgb, model='cnn')

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open('encodings.pickle', 'wb') as encodeFile:
        encodeFile.write(pickle.dumps(data))
