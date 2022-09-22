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

from keras.models import load_model
import numpy as np
import cv2

from smart_assistant.recognition.face import encoder

__author__ = "Steven Kight"
__version__ = "2.0"
__pylint__ = "2.14.4"

with open("smart_assistant/recognition/face/models/names.txt", "rb") as file:
    lines = file.read()
    NAMES = str(lines)
    NAMES = NAMES[2:len(NAMES)-5].split("\\r\\n")
    file.close()

MODEL = load_model("smart_assistant/recognition/face/models/face_model.h5")


def time():
    """
    Gets Current Time for tracking speed of recognition.

    :return: A string of the current time.
    """
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return "Current Time = " + current_time


def run_webcam():
    """
    Runs the webcam until faces are found in the frame

    :return: A frame containing one or more faces within.
    """
    global FACE_CASCADE
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades +
                        "haarcascade_frontalface_default.xml")

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

    print(time(), "- Running Webcam")
    face_frame = run_webcam()
    print(time(), "- After Cam")

    # Convert the image from BGR color to RGB color
    rgb_frame = face_frame[:, :, ::-1]

    face_locations = encoder.face_locations_frame(face_frame)

    if len(face_locations):
        for face in face_locations:

            face_encodings = encoder.face_encodings(rgb_frame, [face])

            np_encodings = np.array(face_encodings)

            res = MODEL.predict(np_encodings, batch_size=len(np_encodings[0]))

            persons = []
            for result in res:
                max_index = np.argmax(result)

                res = res.tolist()[0]
                confidence = res[max_index]

                persons.append((NAMES[max_index], confidence))
            
            return persons
                

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
    print(time(), "- Finish \n")

    for name, conf in names:
        if conf > 0.4:
            print(name, ":", conf)
        elif conf > 0.1 and conf < 0.2:
            print(f"This is more than likely an image of {name} : {conf}")
        else:
            print(f"Unknown, closest is: {name} : {conf}")
