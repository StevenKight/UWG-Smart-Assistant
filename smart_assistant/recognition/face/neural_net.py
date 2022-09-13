"""
Testing Different Binary Classification models on testing
training to learn how to recognize just Me (Steven Kight)

Pylint: 9.88 (August 26, 2022)
E1101 disabled because pylint cannot find cv2 modules
E0401 disabeled because of importing recognition encoder error
"""

# pylint: disable=E0401
# pylint: disable=E1101

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import cv2
import pandas as pd

from smart_assistant.recognition.face import encoder

__author__ = "Steven Kight"
__version__ = "1.2"
__pylint__ = "2.14.4"

NAMES = None


def get_known_info():
    """
    Imports known encodings and names.

    ### Return
        A list of all encodings and a dictionary with names as keys and encodings as values.
    """

    global NAMES

    header = ["0","1","2","3","4","5","6","7","8","9","10","11",
            "12","13","14","15","16","17","18","19","20","21","22",
            "23","24","25","26","27","28","29","30","31","32","33",
            "34","35","36","37","38","39","40","41","42","43","44",
            "45","46","47","48","49","50","51","52","53","54","55",
            "56","57","58","59","60","61","62","63","64","65","66",
            "67","68","69","70","71","72","73","74","75","76","77",
            "78","79","80","81","82","83","84","85","86","87","88",
            "89","90","91","92","93","94","95","96","97","98","99",
            "100","101","102","103","104","105","106","107","108",
            "109","110","111","112","113","114","115","116","117",
            "118","119","120","121","122","123","124","125","126",
            "127","Name"]

    dataframe = pd.read_csv("smart_assistant/recognition/face/models/data.csv",
        names=header)

    labels = dataframe.pop('Name')
    labels.pop(0)
    labels = np.array(labels.values.tolist())
    NAMES = np.unique(labels)

    binary_labels = []
    for person in labels:
        zeros = np.zeros(len(NAMES), dtype=int)
        index = np.where(NAMES == person)[0][0]
        zeros[index] = 1
        binary_labels.append(zeros)

    header.pop(len(header)-1)
    numeric_features = dataframe[header]
    numeric_features = numeric_features.iloc[1: , :]
    numeric_features = numeric_features.values.tolist()

    return numeric_features, binary_labels


def train():
    """
    Uses a list of different Binary Models to test for best accuracy
    across mulitple models to find best one to use for each individual
    """

    x_train, y_train = get_known_info()
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    size_x = len(x_train[0])
    size_y = len(y_train[0])

    model = Sequential()
    model.add(Dense(size_x, input_dim=size_x, activation='tanh', name='hidden_layer'))
    model.add(Dense(units=size_y, activation='softmax', name='output_layer'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print(f'Summary:\n{model.summary()}')
    print(f'Input: {model.input_shape}')
    print("Full shape:" , np.shape(x_train), "Per example shape:" , np.shape(x_train[0]))

    EPOCHS = 10
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=1, verbose=1)
    model.save('smart_assistant/recognition/face/models/face_model.h5', history)

    return model


def run_webcam():
    """
    Runs the webcam until faces are found in the frame

    #### Return
       A frame containing one or more faces within.
    """

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

    ### Return
       A list of the name of all persons within the cameras view.
    """
    face_encodings = []

    face_frame = run_webcam()

    # Convert the image from BGR color to RGB color
    rgb_frame = face_frame[:, :, ::-1]

    face_locations = encoder.face_locations_frame(face_frame)

    if len(face_locations):
        for face in face_locations:

            face_encodings.append(encoder.face_encodings(rgb_frame, [face]))

        return face_encodings


def test(model):
    """
    Utilizes a passed in model or list of models and the webcam
    to predict the person in the frame based on the model.
    """

    data = recognize_person()[0]
    data = np.array(data)

    try:
        res = model.predict(data, batch_size=len(data[0]))
        max_index = np.argmax(res)
        print(NAMES[max_index])
    except ValueError:
        print("ValueError")


if __name__ == "__main__":
    models = train()
    test(models)
