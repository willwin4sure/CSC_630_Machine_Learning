import sys

with open(sys.argv[1], 'r+') as f:
    for idx, line in enumerate(f):
        line = line.lower()
        line = line.strip('\n.')
        words = line.split(' ')
        for i in range(len(words)-1):
            if words[i] == words[i+1]:
                print(f'In line {idx}, the word {words[i]} is repeated at character {i}.')
