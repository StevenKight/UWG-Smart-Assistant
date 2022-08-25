"""
Testing Different Binary Classification models on testing
training to learn how to recognize just Me (Steven Kight)

Pylint: 9.85 (August 25, 2022)
E1101 disabled because pylint cannot find cv2 modules
E0401 disabeled because of importing recognition encoder error
W0601 disabled because global variables are necessary
"""

# pylint: disable=E0401
# pylint: disable=E1101
# pylint: disable=W0601

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from Recognition import encoder

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"

def get_known_info():
    """
    Imports known encodings and names.

    :return: A list of all encodings and a dictionary with names as keys and encodings as values.
    """

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
    numeric_feature_names = ["0","1","2","3","4","5","6","7","8","9","10","11",
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
            "127"]

    dataframe = pd.read_csv("Recognition/Models/NN/data.csv",
        names=header)

    labels = dataframe.pop('Name')
    labels.pop(0)
    labels = labels.values.tolist()
    labels = [int(float(value)) for value in labels]

    numeric_features = dataframe[numeric_feature_names]
    numeric_features = numeric_features.iloc[1: , :]
    numeric_features = numeric_features.values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        numeric_features, labels, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test


def train():
    """
    Uses a list of different Binary Models to test for best accuracy
    across mulitple models to find best one to use for each individual
    """

    x_train, x_test, y_train, y_test = get_known_info()

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    models = []
    scores = []
    for clf in classifiers:
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        models.append(clf)
        scores.append(score)

    print(scores)

    print("models created")

    return models


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
    face_encodings = []

    face_frame = run_webcam()

    # Convert the image from BGR color to RGB color
    rgb_frame = face_frame[:, :, ::-1]

    face_locations = encoder.face_locations_frame(face_frame)

    if len(face_locations):
        for face in face_locations:

            face_encodings.append(encoder.face_encodings(rgb_frame, [face]))

        return face_encodings


def test(models):
    """
    Utilizes a passed in model or list of models and the webcam
    to predict the person in the frame based on the model.
    """

    data = recognize_person()

    res = []
    if isinstance(models, list):
        for model in models:
            res.append(model.predict(data[0][0].reshape(1,-1)))
    else:
        res = model.predict(data)

    return res
