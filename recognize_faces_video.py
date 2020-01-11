from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from ctypes import CDLL
import Quartz


def isUserInFrame(names):
    for person in names:
        if 'me' == person:
            return True
    return False


def recognizeFaces(args, data, vs):
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # TODO change to hog for performance
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        
        # attempt to match each face in the input image
        matches = face_recognition.compare_faces(
            data["encodings"], encoding, tolerance=0.5)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    return frame, names


def isLocked():
    d = Quartz.CGSessionCopyCurrentDictionary()
    # MAC only right now
    # TODO same check for windows
    return bool(d.get('CGSSessionScreenIsLocked'))
