"""
This file takes the json file comprised of patterns and associated tags and creates
machine learning model for the tasks.py file to utilize
"""

import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import layers_list

import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

__author__ = "Steven Kight"
__version__ = "1.3"
__pylint__ = "2.12.2"

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('Conversation/Tasks/Models/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes")

print(len(words), "unique lemmatized words")

pickle.dump(words, open('Conversation/Research/Growth_NN/Models/words.pkl', 'wb'))
pickle.dump(classes, open('Conversation/Research/Growth_NN/Models/classes.pkl', 'wb'))

training = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []

    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

model = Sequential()

""" 
The number of layers, number of nodes per layer, when and how many dropout layers, and dropout rate
need to be variables gotten from another algorithm
"""
layers = layers_list.layer_testing()
network_count = 0

while True:
    for layer in layers:
        if len(layer) == 1:
            continue
        elif layers.index(layer) == 1:
            model.add(Dense(layer[0], input_shape=(len(train_x[0]),), activation='relu'))
        else:
            model.add(Dense(layer[0], activation='relu'))
        
        if layer[1]:
            model.add(Dropout(layer[2]))

    """
    How to get an output?
    """
    model.add(Dense(len(train_y[0]), activation='softmax'))


    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    network_count += 1
    epochs = layers[0][0]

    hist = model.fit(
        np.array(train_x), np.array(train_y), 
        epochs=epochs, batch_size=5, verbose=1
        ) # Validation values would go here to get the validation loss and accuracy

    accuracy_history = hist.history["accuracy"]
    final_accuracry = accuracy_history[len(accuracy_history)-1]

    if final_accuracry >= 0.92:
        break
    else:
        break
        layers = layers_list.layer_testing(layers, final_accuracry)


model.save('Conversation/Research/Growth_NN/Models/chatbot_model.h5', hist)
print("model created")

# Plotting loss and accuracy (Not Neccessary for normal operation)
loss_history = hist.history["loss"]
accuracy_history = hist.history["accuracy"]
epochs = np.arange(1, epochs+1)

plt.plot(epochs, loss_history, label="Loss")
plt.plot(epochs, accuracy_history, label="Accuracy")

plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.show()
