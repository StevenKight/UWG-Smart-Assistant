
"""
The main way to run a full test of the AI.
"""

import cv2 as cv

import smart_assistant.conversation.tasks as tasks
from smart_assistant.recognition import face

def testDevice(source):
    cap = cv.VideoCapture(source) 
    if cap is None or not cap.isOpened():
        return False
    
    cap.release()
    return True
    

def main(preview=False):
    """
    This runs the smart assistant in its entirety
    """

    if preview:
        people = ['Steven Kight']

    else:
        camera = False
        if testDevice(0):
            camera = True

        if not camera:
            print("Audio not implemented")
            people = ['Steven Kight']

            #people = audio.recognize_person()
        else:
            people = face.recognize_person()
            people = [person[0] for person in people]

    tasks.running(people)

if __name__ == "__main__":
    main(True)
    