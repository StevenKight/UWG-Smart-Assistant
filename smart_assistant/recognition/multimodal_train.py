# pylint: disable=E0401
# pylint: disable=E1101

import itertools
import os
import random
import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

__author__ = "Steven Kight"
__version__ = "1.2"
__pylint__ = "2.14.4"

tf.random.set_seed(6)
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


def get_csv_info_face():
    """
    Get the information from the csv.
    """

    dataframe = pd.read_csv("smart_assistant/recognition/face/models/data.csv",
        names=HEADER)

    dataframe_filter = ['Andy Kight', 'Braylin Logan', 'Joey Kight', 'Robin Wright', 'Steven Kight', 'Tracy Kight']
    dataframe = dataframe[dataframe['Name'].isin(dataframe_filter)]

    return dataframe


def get_names_info_face(data):
    """
    Gets the names of all people in the dataset
    """

    labels = data.pop('Name')
    labels = np.array(labels.values.tolist())
    unique = np.unique(labels)

    with open('smart_assistant/recognition/face/models/names_small.txt', 'w', encoding='utf-8') as file:
        for name in unique:
            file.write(f"{name}\n")
        print('Done')

    return labels, unique


def get_known_info_face():
    """
    Imports known encodings and names.

    ### Return
        A list of all encodings and a dictionary with names as keys and encodings as values.
    """

    try:
        os.remove('smart_assistant/recognition/face/models/names_small.txt')
        os.remove('smart_assistant/recognition/face/models/face_model_small.h5')
    except OSError:
        pass

    dataframe = get_csv_info_face()

    labels, uniques = get_names_info_face(dataframe)

    binary_labels = []
    for person in labels:
        zeros = np.zeros(len(uniques), dtype=int)
        index = np.where(uniques == person)[0][0]
        zeros[index] = 1
        binary_labels.append(zeros)

    HEADER.pop(len(HEADER)-1)
    numeric_features = dataframe[HEADER]
    numeric_features = numeric_features.iloc[0: , :]
    numeric_features = numeric_features.values.tolist()

    return numeric_features, binary_labels, uniques


def process_data_audio(file_path):
    rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        audio_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in audio_reader:
            rows.append(row)

    rows.pop(0)

    for index in enumerate(rows,0):
        rows[index[0]] = ''.join(rows[index[0]]).split(',')

    # Label Creation
    names = []
    for row in rows:
        names.append(row.pop(len(row)-1))

    labels = []
    for index in enumerate(names,0):
        label_array = np.zeros(len(names), int)
        np.put(label_array, index[0],1)
        labels.append(label_array)

    # Data Creation
    for i in enumerate(rows,0):
        for j in enumerate(rows[i[0]],0):
            rows[i[0]][j[0]] = float(rows[i[0]][j[0]])

    data = []
    for row in rows:
        data.append(np.array(row))

    return data, labels, names


def load_data_audio():

    data, labels, _ = process_data_audio('smart_assistant/recognition/audio/Models/data.csv')

    # Relation Creation
    training = []
    if len(data) == len(labels):
        for i in enumerate(labels,0):
            training.append((data[i[0]], labels[i[0]]))

    random.shuffle(training)
    training = np.array(training, dtype=object)

    x_train = list(training[:, 0])
    y_train = list(training[:, 1])
    print(f"Training data created with {np.shape(x_train[0])} input shape")

    return x_train, y_train


def proccess_combination_data():
    audio_x, _ = load_data_audio()
    face_x, face_y, _ = get_known_info_face()
    face_data = tuple(zip(face_x, face_y))

    combinations_x = list(itertools.product(audio_x, face_x))
    combinations_flattened = [list(itertools.chain.from_iterable(combination)) for combination in combinations_x]
    combinations_flattened = np.array(combinations_flattened)

    combinations_y = list(itertools.product(audio_x, face_data))
    labels = [tup[1] for tup in combinations_y]
    labels = [tup[1] for tup in labels]

    return combinations_flattened, labels


def train():
    """
    Uses a list of different Binary Models to test for best accuracy
    across mulitple models to find best one to use for each individual
    """

    x_train, y_train = proccess_combination_data()

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    size_x = len(x_train[0])
    size_y = len(y_train[0])

    model = Sequential()
    model.add(Dense(size_x, activation='tanh', name='hidden_layer'))
    model.add(Dense(units=size_y, activation='softmax', name='output_layer'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    epochs = 25
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=1)
    model.save('smart_assistant/recognition/models/multimodel_model.h5', history)

    print(f'Summary:\n{model.summary()}')
    print(f'Input: {model.input_shape}')
    print("Full shape:" , np.shape(x_train), "Per example shape:" , np.shape(x_train[0]))

    return model

if __name__ == "__main__":

    trained_model = train()
