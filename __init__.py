"""
The main way to run a full test of the AI.
"""

import conversation
from recognition import face, audio

if __name__ == "__main__":

    camera = input("camera? (y or n) ")

    if camera in "n":
        camera = False
    else:
        camera = True

    if not camera:
        people = audio.recognize_person()
    else:
        people = face.recognize_person()

    conversation.running(people)
