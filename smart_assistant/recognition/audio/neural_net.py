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

__author__ = "Ana Stanescu"
__version__ = "1.6"
__pylint__ = "2.14.4"


X_TRAIN = 'will hold train'

def process_data(file_path):
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

    return data, labels


def load_data():

    data, labels = process_data('smart_assistant/research/audio/Models/data.csv')

    # Relation Creation
    training = []
    if len(data) == len(labels):
        for i in enumerate(labels,0):
            training.append((data[i[0]], labels[i[0]]))

    random.shuffle(training)
    training = np.array(training)

    x_train = list(training[:, 0])
    y_train = list(training[:, 1])
    print(f"Training data created with {np.shape(x_train[0])} input shape")
    return x_train, y_train

def train():
    x_train, y_train = load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(f'x_train:\n{x_train}')
    print(f'x_train:\n{y_train}')
    X_TRAIN = x_train

    model = Sequential()
    model.add(Dense(1, input_dim=len(x_train[0]), activation='tanh', name='hidden_layer'))
    #model.add(Dense(64, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='softmax', name='output_layer'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(f'Summary:\n{model.summary()}')

    EPOCHS = 2
    history = model.fit(X_TRAIN, y_train, epochs=EPOCHS, verbose=1)    
    #model.save('Research/Audio/Models/audio_model.h5', hist)
    print("Full shape:" , np.shape(X_TRAIN), "Per example shape:" , np.shape(x_train[0]))
    return model


def test(model):
    data, _ = process_data('smart_assistant/research/audio/Models/test.csv')
    print(f'data[0] {data[0]}')
    print(np.shape(data[0]))
    print(data[0])

    prediction_vector = model.predict([[3299.73,227.14,4159.46,18.03,428.79,\
        3272.01,3327.45,427.32,1372.11,4.42,25.58,420.85,433.78]])
    print(prediction_vector)
    print(f'index: {np.argmax(prediction_vector)}')

if __name__ == "__main__":
    model = train()
    test(model)
