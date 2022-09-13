"""
This code is the main function of the facial recognition.

This code utilizes the computers built in webcam to find faces within the view and then
passes the image of those faces to an encoder from the modual face_recognition and then compares
the encoding with known encodings from the saved lists created by the Encoding.py file and finds
the best match utilizing a distance determination again by the module face_recogniton.

Note: Look into pylint error W0601

A portion of this code comes from:
https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
Other parts of it come from:
https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
All other parts were written by me

Pylint: 9.73 (August 25, 2022)
E1101 disabled because pylint cannot find cv2 modules
E0401 disabeled because of importing recognition encoder error
"""

# pylint: disable=E1101
# pylint: disable=E0401
# pylint: disable=W0601

import pickle
import os
import shutil
import uuid
from datetime import datetime

import numpy as np
import cv2

from smart_assistant.recognition.face import encoder

__author__ = "Steven Kight"
__version__ = "2.0"
__pylint__ = "2.14.4"

def time():
    """
    Gets Current Time for tracking speed of recognition.

    :return: A string of the current time.
    """
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return "Current Time = " + current_time

def get_known_info():
    """
    Imports known encodings and names.

    :return: A list of all encodings and a dictionary with names as keys and encodings as values.
    """
    global NAME_ENCODINGS
    global FACE_ENCODINGS

    # Create arrays of known face encodings and their names
    with open("smart_assistant/recognition/face/models/Encodings.txt", "rb") as encoded:
        FACE_ENCODINGS = pickle.load(encoded)
        encoded.close()

    with open("smart_assistant/recognition/face/models/Encodings_Names.txt", "rb") as encoded_names:
        NAME_ENCODINGS = pickle.load(encoded_names)
        encoded_names.close()

    return FACE_ENCODINGS, NAME_ENCODINGS

def run_webcam():
    """
    Runs the webcam until faces are found in the frame

    :return: A frame containing one or more faces within.
    """
    global FACE_CASCADE
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades +
                        "haarcascade_frontalface_default.xml")

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        frame = video_capture.read()[1]

        face_frame = encoder.face_locations_frame(frame)

        if len(face_frame):
            break

    # Release handle to the webcam
    video_capture.release()

    return frame

def recognize_person():
    """
    Takes a frame and utilizes it and lists of known faces
    to find the persons name.

    :return: A list of the name of all persons within the cameras view.
    """
    face_names = []

    get_known_info()

    face_frame = run_webcam()

    # Convert the image from BGR color to RGB color
    rgb_frame = face_frame[:, :, ::-1]

    face_locations = encoder.face_locations_frame(face_frame)

    if len(face_locations):
        for face in face_locations:

            face_encodings = encoder.face_encodings(rgb_frame, [face])

            for face_encoding in face_encodings:
                # Find the known face with the smallest distance to the new face
                face_distances = encoder.face_distance(FACE_ENCODINGS, face_encoding)
                best_match_index = np.argmin(face_distances)

                match = FACE_ENCODINGS[best_match_index]

                # Go through dictionary with each face and the name of the directory
                for key in NAME_ENCODINGS.keys():

                    # Get the list of encodings associated with that person
                    values = NAME_ENCODINGS.get(key)
                    # Cycle through each encoding
                    for value in values:
                        # Check if it is the same as the match and if it is save the key
                        if np.array_equal(value, match):
                            name_key = key

                face_names.append(name_key)

        # Go through each person recognized
        #for face in face_locations:
        #    save_new_image(face_frame, face, name_key)

        return face_names

def save_new_image(image, points, img_name):
    """
    Crops frame to zoom to face and save it to that persons file.

    :param image: The frame containing the face of a person.
    :param points: A list of points from dlib 5-point pose estimator.
    :param img_name: The name of the person in the image.
    """

    # All the images are in one folder named "Dataset Images".
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset')

    # Then Save the image then move it to that persons directory
    new_file_location = path + "/" + img_name + "/Images/"
    new_file_name = str(uuid.uuid4())+'.jpg'

    cropped = image[points[0]:points[2], points[3]:points[1]]
    cv2.imwrite(new_file_name, cropped)

    shutil.move(new_file_name, new_file_location + new_file_name)

    cv2.destroyAllWindows()

if __name__ == "__main__":

    print(time(), "- Start")
    names = recognize_person()
    print(time(), "- Finish")

    INTRODUCTION = ""
    for index, name in enumerate(names):
        if index == 0:
            INTRODUCTION += "Hello " + name
        else:
            INTRODUCTION += " and " + name

    print(INTRODUCTION)

    #new_encodings.encode_directorys()
