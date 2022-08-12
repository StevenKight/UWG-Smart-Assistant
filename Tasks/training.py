"""
This file takes the json file comprised of patterns and associated tags and creates
machine learning model for the main Tasks file to utilize.

Current Model Accuracy: 0.9659 (August 11, 2022)

Pylint: 9.23 (August 11, 2022)
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

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('Tasks/Models/intents.json').read()
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

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('Tasks/Models/words.pkl', 'wb'))
pickle.dump(classes, open('Tasks/Models/classes.pkl', 'wb'))

training = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []

    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

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

EPOCHS = 100
hist = model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, batch_size=5, verbose=1)
model.save('Tasks/Models/chatbot_model.h5', hist)

accuracy_history = hist.history["loss"]
numpy_accuracy_history = np.array(accuracy_history)
final_accuracy = accuracy_history[EPOCHS-1]

print("model created")
