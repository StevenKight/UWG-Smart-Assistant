"""
The main way to run a full test of the AI.
"""

import conversation
import recognition
from recognition import encodings

if __name__ == "__main__":

    camera = input("camera? (y or n) ")

    if camera in "n":
        camera = False
    else:
        camera = True

    if not camera:
        people = ["Steven Kight"]
    else:
        people = recognition.recognize_person()

    conversation.running(["Steven Kight"])
