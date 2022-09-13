"""
Testing Different Binary Classification models on testing
training to learn how to recognize just Me (Steven Kight)

Pylint: 9.88 (August 26, 2022)
E1101 disabled because pylint cannot find cv2 modules
E0401 disabeled because of importing recognition encoder error
"""

# pylint: disable=E0401
# pylint: disable=E1101

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import cv2
import pandas as pd

from smart_assistant.recognition.face import encoder

__author__ = "Steven Kight"
__version__ = "1.2"
__pylint__ = "2.14.4"

NAMES = None

HEADER = ["0","1","2","3","4","5","6","7","8","9","10","11",
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
            "127", "Name"]


def load_unique_names():
    """
    Gets the list of names saved in a .txt.
    """

    global NAMES
    
    names = []

    with open('smart_assistant/recognition/face/models/names.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            x = line[:-1]
            names.append(x)

    NAMES = names


def get_csv_info():
    """
    Get the information from the csv.
    """
 
    dataframe = pd.read_csv("smart_assistant/recognition/face/models/data.csv",
        names=HEADER)

    return dataframe


def get_names_info(data):
    """
    Gets the names of all people in the dataset
    """

    global NAMES

    labels = data.pop('Name')
    labels.pop(0)
    labels = np.array(labels.values.tolist())
    NAMES = np.unique(labels)

    with open('smart_assistant/recognition/face/models/names.txt', 'w', encoding='utf-8') as file:
        for name in NAMES:
            file.write(f"{name}\n")
        print('Done')

    return labels


def get_known_info():
    """
    Imports known encodings and names.

    ### Return
        A list of all encodings and a dictionary with names as keys and encodings as values.
    """

    try:
        os.remove('smart_assistant/recognition/face/models/names.txt')
        os.remove('smart_assistant/recognition/face/models/face_model.h5')
    except OSError:
        pass

    dataframe = get_csv_info()

    labels = get_names_info(dataframe)

    binary_labels = []
    for person in labels:
        zeros = np.zeros(len(NAMES), dtype=int)
        index = np.where(NAMES == person)[0][0]
        zeros[index] = 1
        binary_labels.append(zeros)

    HEADER.pop(len(HEADER)-1)
    numeric_features = dataframe[HEADER]
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


def test(test_model):
    """
    Utilizes a passed in model or list of models and the webcam
    to predict the person in the frame based on the model.
    """

    data = recognize_person()

    for item in data:
        item = np.array(item)

        try:
            res = test_model.predict(item, batch_size=len(item[0]))
            max_index = np.argmax(res)

            res = res.tolist()[0]
            confidence = res[max_index]

            if confidence > 0.5:
                print(NAMES[max_index], ":", res[max_index])
            else:
                print(f"Unknown, closest is: {NAMES[max_index]} : {res[max_index]}")

        except ValueError:
            print("ValueError")


if __name__ == "__main__":
    try:
        testing_model = load_model('smart_assistant/recognition/face/models/face_model.h5')
        load_unique_names()
    except OSError:
        testing_model = train()

    test(testing_model)
