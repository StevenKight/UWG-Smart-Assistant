import glob
import os

os.chdir('Research/Language/Grammar/Words')
files = glob.glob('*.txt')

for file in files:
    a_file = open(file, "r")

    list_of_lists = [line.strip().lower() for line in a_file][1:]

    a_file.close()

    print("' | '".join(list_of_lists))
    print(file)

    if input(": "):
        continue