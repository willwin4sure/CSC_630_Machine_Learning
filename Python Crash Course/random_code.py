'''
Below is a program designed to estimate the answer to the following problem by repeated simulation:
    A standard 52-card deck is shuffled, and the cards are turned over one-at-a-time
    starting with the top card. What is the expected number of cards that will be turned over
    at the time when we see the first Ace?

Note that there is a simple (but nonobvious) mathematical way to solve this problem, by simply noting that
the four aces divide the remaining 48 cards into 5 equal sections, and in expectation each section should have
the same number of non-ace cards: therefore, the first section has in expectation 48/5 cards. 
We add the ace that we draw to get 48/5+1=10.6.
'''

import random

def drawing_aces(n = 100000):
    total = 0
    for _ in range(n):
        for i in range(52):
            if (random.randint(1,52-i) < 5): # if card is an ace
                total += i+1
                break
    return total/n

print(drawing_aces())
