"""
This file is functions that allow for a simplified way to
modify the dataset as a whole.
Pylint: 10.00 (August 11, 2022)
"""

import os
import shutil

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"


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


def new_person(name, image="", voice=""):
    """
    Adds a new person to the Dataset/ directory whose
    name is the inputed value.

    ### Parameters
        `name` (`str`): A String representing the new persons name.
        `image` (`str`): A String representation of the new persons face image.
        `voice` (`str`): A String representation of the new persons voice recording.
    """

    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Dataset/')
    new_dir = path + name

    # Make new directory
    os.mkdir(new_dir)

    # Include images dir and voice dir
    os.mkdir(new_dir + "/Images")
    if image != "":
        shutil.move(image, new_dir + "/Images" + image)
    os.mkdir(new_dir + "/Voice")
    if voice != "":
        shutil.move(voice, new_dir + "/Voice" + voice)


if __name__ == "__main__":
    print("Modifying Dataset")

    # Place modifications on next line


    print("Modifications Complete")
