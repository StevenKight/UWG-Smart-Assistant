import random
import json
import pickle

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def sentence_to_ascii(sentence: str):

    sentence_list = []
    for word in sentence.split():
        sentence_list.append([ord(letter) for letter in word])
        sentence_list.append([32])

    # Flatten the list of lists
    sentence_list = [item for sublist in sentence_list for item in sublist]

    return sentence_list

def ascii_to_sentence(ascii_list: list):
    return ''.join(chr(i) for i in ascii_list)

def patterns_testing():
    data_file = open('Conversation/Tasks/Models/intents.json').read()
    intents = json.loads(data_file)

    patterns = []
    labels = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(sentence_to_ascii(pattern))
            labels.append(sentence_to_ascii(intent['tag']))

    return patterns, labels

def standard_training():
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
    return train_x, train_y

patterns = patterns_testing()
standard = standard_training()
print("New:", len(patterns[0]), "Standard:", len(standard[0]))
print("New:", len(patterns[1]), "Standard:", len(standard[1]))
