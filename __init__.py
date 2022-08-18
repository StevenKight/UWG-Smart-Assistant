"""
The main way to run a full test of the AI.
"""

import Tasks
import Recognition

if __name__ == "__main__":
    # Special Check for presentation
    camera = input("camera? (y or n) ")

    if camera == "n":
        camera = False
    camera = True
    
    if not camera:
        people = ["Steven Kight"]
    else:
        people = Recognition.recognize_person()

    Tasks.running(people)
