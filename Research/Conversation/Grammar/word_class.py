import csv
import glob
import os

os.chdir('Conversation/Research/Grammar/Words')
files = glob.glob('*.txt')

words_file = open('/Users/andykight/Documents/GitHub/April/Conversation/Research/Grammar/3000 words.csv')
csvreader = csv.reader(words_file)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)
print(f'{len(rows)} Words gathered')

list_of_words = []
for file in files:
    a_file = open(file, "r")

    list_of_lists = [line.strip().lower() for line in a_file]

    a_file.close()
    
    for value in list_of_lists:
        list_of_words.append(value)

print(f'{len(list_of_words)} Words in parts of speech')

for word in list_of_words:
    try:
        rows.remove(word)
    except ValueError:
        pass
    
print(len(rows))
