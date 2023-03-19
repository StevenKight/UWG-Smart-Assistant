import csv
import itertools

import numpy as np

def toBinary(a):
  l = []
  m = ''
  for i in a:
    l.append(ord(i))
  for i in l:
    m += bin(i)[2:]
  return m

ENDINGS = [toBinary('.'), toBinary('?'), toBinary('!')]

def BinaryToDecimal(binary):
        
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
        
    return (decimal)   


def BinaryToString(binary):
    if type(binary) == list:
        binary = ListToString(binary)
    
    str_data =' '
    for j in range(0, len(binary), 7):
        temp_data = int(binary[j:j + 7])
        decimal_data = BinaryToDecimal(temp_data)
        str_data = str_data + chr(decimal_data)

    return str_data

def ListToString(list):
    string = "" 
    
    for ele in list:
        string += str(ele)  
    
    return string

def transition():
    """ Calculates the transition or Q value of each state """

    q_function = (1 - alpha) * state_q + alpha * (reward + gamma * future_q_values[max_index])

    return q_function, future_key

def epoch(table):
    path = []
    start_word = table[0][np.random.randint(0, len(table[0]))]
    path.append(start_word)

    for col_val in range(len(table)):
        next = table[col_val][np.random.randint(0, len(table[0]))]
        next = next.removeprefix('[').removesuffix(']').split(',')
        path.append(next)

        if next in ENDINGS:
            break

    return path

def run(epochs):

    information = []

    with open('Conversation/Research/Sentence Building/Table.csv', mode='r') as inp:
        reader = csv.reader(inp)
        table = list(reader)

    counter = itertools.count(1, 1)
    while counter.__next__() <= epochs:
        sentence = epoch(table)
        sentence[0] = sentence[0].removeprefix('[').removesuffix(']').split(',')

        for word_index in range(len(sentence)):

            try:
                sentence[word_index] = [int(element) for element in sentence[word_index]]
            except:
                pass

        string = ''
        for i in sentence:
            string += BinaryToString(i)

        if string[0] == ' ':
            string = string.removeprefix(' ')
        elif '  ' in string:
            string = string.replace('  ', ' ')

        list_sentence = string.split(' ')

        information.append(f'Words/punctuation: {len(list_sentence)}, Final word/punctuation: {list_sentence[len(list_sentence)-1]}')

    return information[0], list_sentence
