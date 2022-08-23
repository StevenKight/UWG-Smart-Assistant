import math
import csv

MAX_COLUMNS = 10

words_file = open('Conversation/Research/Sentence Building/3000 words.csv')

csvreader = csv.reader(words_file)

header = []
header = next(csvreader)

def toBinary(a):
  l = []
  m = ''
  for i in a:
    l.append(ord(i))
  for i in l:
    m += bin(i)[2:]
  return m

rows = []
for row in csvreader:
    rows.append(toBinary(row[0]))
print(f'{len(rows)} Words gathered')

table = [] 
column_index = 0
while column_index < MAX_COLUMNS:
    table.append(rows)
    column_index += 1
print(f'{MAX_COLUMNS} Columns generated')

with open("Conversation/Research/Sentence Building/Table.csv", "w") as file_t:
    t = csv.writer(file_t)
    t.writerows(table)

print('Table saved')

"""
for index in range(len(table)):
    q_table = {}
    if index != len(table)-1:
        column = table[index]
        next_column = table[index+1]
    
    for row in column:
        for next_row in next_column:
            key = (row, next_row)
            q_table[key] = 0
    
    print(f'Column {index+1} complete out of {len(table)} initialized')

print('Q-Table initialized')

with open('Conversation/Research/Sentence Building/Q-Table.csv', 'w') as f:
    for key in q_table.keys():
        f.write("%s,%s\n"%(key,q_table[key]))

print('Q-Table saved')
"""
