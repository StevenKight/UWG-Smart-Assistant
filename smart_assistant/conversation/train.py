"""
This file takes the json file comprised of patterns and associated tags and creates
machine learning model for the main conversation file to utilize.

Current Model Accuracy: 0.9659 (August 11, 2022)

Pylint: 9.55 (August 25, 2022)
"""

import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout

lemmatizer = WordNetLemmatizer()

__author__ = "Steven Kight"
__version__ = "1.5"
__pylint__ = "2.14.4"

def data_processing():
    """
    Utilizes the intentes.json file in the models directory to
    create necessary data in the necessary shapes to be utilized
    in a training Neural Network.
    """

    words = []
    classes = []
    documents = []
    data_file = open('smart_assistant/conversation/models/intents.json', encoding='UTF-8').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)

            documents.append((word, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])


    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!']]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")

    print(len(classes), "classes")

    print(len(words), "unique lemmatized words")

    pickle.dump(words, open('smart_assistant/conversation/models/words.pkl', 'wb'))
    pickle.dump(classes, open('smart_assistant/conversation/models/classes.pkl', 'wb'))

    training = []

    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []

        pattern_words = doc[0]

        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        for word in words:
            if word in pattern_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    print("Training data created")

    return list(training[:, 0]), list(training[:, 1])

def train_model():
    """
    Utilizes the preprocessed data from `data_processing()` to
    train a Neural Network model on how to recognize the patterns
    in the tags held within the intents.json file in the models
    directory.
    """

    train_x, train_y = data_processing()
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    epochs = 100
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=5, verbose=1)
    model.save('smart_assistant/conversation/models/chatbot_model.h5', history)

    print("model created")

if __name__ == "__main__":
    train_model()
