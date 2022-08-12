"""
The main way to run a full test of the AI.
"""

import Tasks
import Recognition

if __name__ == "__main__":
    # Special Check for presentation
    PRESENTATION = input("Presentation? (y or n) ")

    if PRESENTATION == "y":
        PRESENTATION = True
    elif PRESENTATION == "n":
        PRESENTATION = False
    if PRESENTATION:
        people = ["Steven Kight"]
    else:
        people = Recognition.recognize_person()

    Tasks.running(people)
