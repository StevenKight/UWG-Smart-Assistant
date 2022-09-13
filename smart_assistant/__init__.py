
"""
The main way to run a full test of the AI.
"""

import smart_assistant.conversation as tasks
from smart_assistant.recognition import face, audio

def main():
    """
    This runs the smart assistant in its entirety
    """

    camera = input("camera? (y or n) ")

    if camera in "n":
        camera = False
    else:
        camera = True

    if not camera:
        print("Audio not implemented")
        people = ['Steven Kight']

        #people = audio.recognize_person()
    else:
        people = face.recognize_person()

    tasks.running(people)

if __name__ == "__main__":
    main()
    