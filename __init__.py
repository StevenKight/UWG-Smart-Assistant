"""
The main way to run a full test of the AI.
"""

import Tasks
import Recognition

if __name__ == "__main__":
    # Special Check for camera
    camera = input("camera? (y or n) ")

    if camera in "n":
        camera = False
    else:
        camera = True
    
    if not camera:
        people = ["Steven Kight"]
    else:
        people = Recognition.recognize_person()
        
    Tasks.running(people)
