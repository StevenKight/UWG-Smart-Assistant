# This folder is for work looking into building and creating grammatically correct sentences.

## Grammer folder:

This folder is for using nltk to check parts of speech and to create a graph that allows the code to understand and tell if a sentence is built properly according to rules set up in .cfg file.

## Growth NN folder:
This folder is used for attempting to create a new style of neural network that can grow and add layers and alter dropouts and multiple parameters of each layer in an attempt to "grow" a functional neural network for understanding problems.

Currently, the NN_training.py file functions with a single list of lists as a way to alter layer count, dropout layer count, nodes per layer, and dropout rates, this list of lists needs to now be used in a training of another algorithm to alter these variables to get more accurate at its predections.

## Sentence Building folder:

This folder is set up like a Q-Learning style of algorithm without the learning currently. It uses a table set up using the 3000 most common words and some puntuation to build a random sentence.

## Goal:
### Use these two builds to create an agent that can fully build its own sentences and potentially know when to build different kinds of sentences.