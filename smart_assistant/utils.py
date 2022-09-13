"""
This file is functions that allow for a simplified way to
modify the dataset as a whole.
Pylint: 10.00 (August 11, 2022)
E1101 Disabled due to pylint not reading open-cv properly
E0401 Disabled due to pylint not reading Recognition file properly
"""

# pylint: disable=E1101
# pylint: disable=E0401

import os
import shutil
import uuid
import wave

import pyaudio
import cv2

import recognition.face as recognition

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"


def get_face_frame(name: str):
    """
    Uses the webcam to get the frame of which a face is contained then
    saves the frame to the persons (`name`) file.

    ### Parameter
        `name` (str): The name of the person in the frame.
    """

    face_frame = recognition.run_webcam()

    face_locations = recognition.encoder.face_locations_frame(face_frame)

    save_new_image(face_frame, face_locations, name)


def save_new_image(image, points, img_name):
    """
    Crops frame to zoom to face and save it to that persons file.

    :param image: The frame containing the face of a person.
    :param points: A list of points from dlib 5-point pose estimator.
    :param img_name: The name of the person in the image.
    """

    # All the images are in one folder named "Dataset Images".
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset')

    # Then Save the image then move it to that persons directory
    new_file_location = path + "/" + img_name + "/Images/"
    new_file_name = str(uuid.uuid4())+'.jpg'

    cropped = image[points[0]:points[2], points[3]:points[1]]
    cv2.imwrite(new_file_name, cropped)

    shutil.move(new_file_name, new_file_location + new_file_name)

    cv2.destroyAllWindows()


def record_voice(name: str):
    """
    Records and saves a sound clip from the default microphone
    to a persons file using the name parameter.

    ### Parameter
        `name` (str): The name of the person whos voice is recorded.
    """

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    # All the images are in one folder named "Dataset Images".
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset')

    # Then Save the image then move it to that persons directory
    filename = path + "/" + name + "/Voice/"
    filename += str(uuid.uuid4())+'.jpg'

    port_audio = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = port_audio.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    port_audio.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(port_audio.get_sample_size(sample_format))
    wave_file.setframerate(fs)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()


def new_directory(dir_name):
    """
    Creates a new Directory called the inputed value
    for every person in Dataset/

    ### Parameters
        `dir_name` (`str`): A String name for the newly created directory.
    """

    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset/')

    # Folders must be named after person
    list_of_folders = next(os.walk(path))[1]

    # Folder names are listed and assigned to “names” variable.
    names = list_of_folders.copy()

    # go through all the directorys name by name
    for name in names:
        dir_path = path + name + "/" + dir_name

        # check if directory exists or not yet
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def new_person(name="", image="", voice=""):
    """
    Adds a new person to the Dataset/ directory whose
    name is the inputed value.

    ### Parameters
        `name` (`str`): A String representing the new persons name.
        `image` (`str`): A String representation of the new persons face image.
        `voice` (`str`): A String representation of the new persons voice recording.
    """

    if name == "":
        name = input("What is the name of the person")

    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset/')
    new_dir = path + name

    # Make new directory
    os.mkdir(new_dir)

    # Include images dir and voice dir
    os.mkdir(new_dir + "/Images")
    if image != "":
        shutil.move(image, new_dir + "/Images" + image)
    else:
        inp = input("Would you like to add a face?")
        if "y" in inp:
            get_face_frame(name)
        else:
            print("Okay")

    os.mkdir(new_dir + "/Voice")
    if voice != "":
        shutil.move(voice, new_dir + "/Voice" + voice)
    else:
        inp = input("Would you like to add a voice?")
        if "y" in inp:
            record_voice(name)
        else:
            print("Okay")


if __name__ == "__main__":
    print("Modifying Dataset")

    # Place modifications on next line
    new_person()

    print("Modifications Complete")
