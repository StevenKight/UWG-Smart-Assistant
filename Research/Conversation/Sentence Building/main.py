"""
Sets hyperparameters and runs the main module
"""

import sys

import module

__author__ = "Steven Kight"
__version__ = "3/13/22"
__pylint__ = "2.12.2"

EPOCHS = 1

if __name__ == "__main__":
    info, sentence = module.run(EPOCHS)

    print(info)
    sentence = " ".join(sentence)
    print(sentence)
