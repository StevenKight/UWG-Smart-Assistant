"""
This file takes the json file comprised of patterns and associated tags and creates
machine learning model for the tasks.py file to utilize

Pylint: 9.23 (August 11, 2022)
"""

import random
import csv

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

__author__ = "Steven Kight"
__version__ = "1.5"
__pylint__ = "2.14.4"


def process_data(file_path):
    rows = []
    with open(file_path, newline='') as csvfile:
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

    return data, labels


def load_data():

    data, labels = process_data('Research/Audio/Models/data.csv')

    # Relation Creation
    training = []
    if len(data) == len(labels):
        for i in enumerate(labels,0):
            training.append((data[i[0]], labels[i[0]]))

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print(f"Training data created with {np.shape(train_x[0])} input shape")
    return train_x, train_y

def train():
    train_x, train_y = load_data()

    model = Sequential()
    model.add(Dense(128, input_shape=np.shape(train_x[0]), activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    EPOCHS = 75
    model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, verbose=1)
    
    #model.save('Research/Audio/Models/audio_model.h5', hist)

    print("model created")
    print("Full shape:" , np.shape(train_x), "Per example shape:" , np.shape(train_x[0]))
    return model


def test(model):

    data, _ = process_data('Research/Audio/Models/test.csv')

    print(np.shape(data[0]))
    print(data[0].tolist())
    model.predict(data[0])

if __name__ == "__main__":
    
    model = train()

    test(model)
