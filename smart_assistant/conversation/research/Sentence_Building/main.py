"""
Sets hyperparameters and runs the main module
"""

import matplotlib.pyplot as plt

import module

__author__ = "Steven Kight"
__version__ = "3/13/22"
__pylint__ = "2.12.2"

MAX_LENGTH = 15
EPOCHS = 1

if __name__ == "__main__":

    sentences = module.run(EPOCHS, MAX_LENGTH)

    for sentence in sentences:
        print(' '.join(sentence))
