import csv

import numpy as np

ENDINGS = ['.', '?', '!']

words_file = open('smart_assistant/conversation/research/Sentence Building/3000 words.csv')

csvreader = csv.reader(words_file)

header = []
header = next(csvreader)

WORDS = []
for row in csvreader:
    WORDS.append(row)


def epoch(sentence_length):
    path = []
    
    for _ in range(sentence_length):
        start_word = WORDS[np.random.randint(0, len(WORDS))][0]
        path.append(start_word)

        if path[-1] in ENDINGS:
            break

    return path


def run(epochs, sentence_length):
    sentences = []

    for _ in range(epochs):
        sentence = epoch(sentence_length)
        sentences.append(sentence)

    return sentences
