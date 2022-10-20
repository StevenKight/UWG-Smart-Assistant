"""
This file takes the json file comprised of patterns and associated tags and creates
machine learning model for the tasks.py file to utilize

Pylint: 9.23 (August 11, 2022)
"""

import random
import csv

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf

__author__ = "Ana Stanescu"
__version__ = "1.6"
__pylint__ = "2.14.4"

tf.random.set_seed(6)

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

    return data, labels, names


def load_data():

    data, labels, _ = process_data('smart_assistant/recognition/audio/Models/data.csv')

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

    nodes_input = len(x_train[0])-1
    hidden_nodes = len(y_train[0])*2
    class_count = len(y_train[0])

    model = Sequential()
    model.add(Dense(nodes_input, input_dim=len(x_train[0]), activation='relu', name='input_layer'))
    model.add(Dense(hidden_nodes, activation='relu', name='hidden_layer'))
    model.add(Dense(class_count, activation='softmax', name='output_layer'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(f'Summary:\n{model.summary()}')

    EPOCHS = 50

    loov = LeaveOneOut()

    test_data = [[],[]]
    for train_index, test_index in loov.split(x_train):
        train_X, test_X = x_train[train_index], x_train[test_index]
        train_Y, test_Y = y_train[train_index], y_train[test_index]

        history = model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=1, verbose=1)

        test_loss, test_acc = model.evaluate(test_X, test_Y)
        test_data[0].append(test_acc)
        test_data[1].append(test_loss)

    print('\nTest accuracy:', test_data[0], '\nTest loss:', test_data[1])

    model.save('smart_assistant/recognition/audio/Models/audio_model.h5', history)
    print("Full shape:" , np.shape(x_train), "Per example shape:" , np.shape(x_train[0]))
    return model


def test(model):
    data, _, names = process_data('smart_assistant/recognition/audio/Models/data.csv')
    print(f"Predict Shape: {np.shape(data[0])}")

    # TODO get prediction from microphone
    prediction_vector = model.predict([[3299.73,227.14,4159.46,18.03,428.79,\
        3272.01,3327.45,427.32,1372.11,4.42,25.58,420.85,433.78]])

    max_index = np.argmax(prediction_vector)
    prediction_vector = prediction_vector.tolist()[0]

    if prediction_vector[max_index] > 0.75:
        print(f'\nIndex: {max_index}, Confidence: {prediction_vector[max_index]}, Name: {names[max_index]}')
    else:
        print(f'\nUnknown, Closest is:\nIndex: {max_index}, Confidence: {prediction_vector[max_index]}, Name: {names[max_index]}')

if __name__ == "__main__":
    try:
        testing_model = load_model('smart_assistant/recognition/audio/Models/audio_model.h5')
    except OSError:
        testing_model = train()

    test(testing_model)
