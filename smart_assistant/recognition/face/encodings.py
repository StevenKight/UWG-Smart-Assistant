
"""
This file is used to get and save the encodings to all faces in the dataset.

Pylint: 10.00 (August 25, 2022)
E0401 disabeled because of importing recognition encoder error
W0601 disabled because global variables are necessary
"""

# pylint: disable=E0401
# pylint: disable=W0601

import csv
import os
from datetime import datetime

import numpy as np

from smart_assistant.recognition.face import encoder

__author__ = "Steven Kight"
__version__ = "1.5"
__pylint__ = "2.14.4"


def time():
    """
    Gets Current Time for tracking speed of recognition.

    :return: A string of the current time.
    """
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return "Current Time = " + current_time


def encode_files(folder_path):
    """
    Encodes images from the directory.

    :param folder_path: The path to the directory that conatins the images to encode.
    :return: A list of encodings for the faces in each image within the directory.
    """
    folder_path += "/Images"

    files_list = list(os.listdir(folder_path))
    number_files = len(files_list)

    list_encoding = []

    for value in range(number_files):
        if files_list[value] == ".DS_Store":
            continue

        file_path = folder_path + "/" + files_list[value]

        image_number = f'image{value}'
        image_encoding = f'image_encoding_{value}'

        # Load a sample picture and learn how to recognize it.
        globals()[image_number] = encoder.load_image_file(file_path)

        try:
            globals()[image_encoding] = encoder.face_encodings(globals()[image_number])[0]
        except IndexError:
            dataset_path = "Dataset/"
            error = folder_path.replace(dataset_path, "")
            error = error.replace("/Images", "")
            print(error + "  -  " + files_list[value])

        encoding = globals()[f'image_encoding_{value}']

        # Store Encoding
        FACE_ENCODINGS.append(encoding)
        list_encoding.append(encoding)

    return list_encoding


def encode_directorys():
    """
    Encodes the face in all images in the directory given.
    """
    # All the images are in one folder named "Dataset".
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset/')

    # Folders must be named after person
    list_of_folders = next(os.walk(path))[1]

    # Folder names are listed and assigned to “names” variable.
    names = list_of_folders.copy()

    # Create arrays for known face encodings and their names
    known_encodings_names = {}
    global FACE_ENCODINGS
    FACE_ENCODINGS = []

    # go through all the directorys name by name
    for name in names:

        directory = path + name
        encoded_list = encode_files(directory)
        known_encodings_names[name] = encoded_list

    save(known_encodings_names)


def save(names_encodings):
    """
    Saves the encodings and the dictionary of known names and encodings.
    """

    labels = []
    for key in list(names_encodings.keys()):
        for i, _ in enumerate(names_encodings[key]):
            labels.append(key)

    csv_list = FACE_ENCODINGS
    for i, _ in enumerate(csv_list):
        csv_list[i] = np.append(csv_list[i], labels[i])

    headers = []
    for i in range(128):
        headers.append(i)
    headers.append("Name")

    with open('smart_assistant/recognition/face/models/data.csv', 'w', encoding='UTF-8') as file:
        # using csv.writer method from CSV package
        write = csv.writer(file)
        write.writerow(headers)
        write.writerows(csv_list)


if __name__ == "__main__":

    print(time(), "- Start")
    encode_directorys()
    print(time(), "- Finish")
