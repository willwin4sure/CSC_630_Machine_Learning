import numpy as np
import random
from statistics import Counter
import matplotlib.pyplot as plt

class VitaminJar():
    def __init__(self, jar_size, num_reds, num_blues, sample_size, sample_size_stdev, percentage_choose_red):
        assert num_reds + num_blues <= jar_size
        
        self.jar_size = jar_size
        self.num_reds = num_reds
        self.num_blues = num_blues
        self.sample_size = sample_size
        self.sample_size_stdev = sample_size_stdev
        self.percentage_choose_red = percentage_choose_red
        self.number_of_reds_each_day = [self.num_reds]

    def progress_a_day(self):
        num_drawn = int(round(*np.random.normal(self.sample_size,self.sample_size_stdev,1)))
        selection = Counter()
        for _ in range(num_drawn):
            if (random.randint(1, self.num_reds+self.num_blues) <= self.num_reds):
                selection['red'] += 1
            else:
                selection['blue'] += 1
        
        if (selection['red'] > 0):
            if (random.random() > self.percentage_choose_red/100):
                self.num_blues -= 1
            else:
                self.num_reds -= 1
        else:
            self.num_blues -= 1

        self.number_of_reds_each_day.append(self.num_reds)

    def exhaust_reds(self):
        while (self.num_reds > 0):
            self.progress_a_day()

if __name__ == '__main__':
    amount_time_taken = []
    for _ in range(1000):
        test_jar = VitaminJar(500,250,250,8,1.3,90)
        test_jar.exhaust_reds()
        amount_time_taken.append(len(test_jar.number_of_reds_each_day))

    amount_time_taken.sort()
    print((amount_time_taken[(len(amount_time_taken)-1)//2]+amount_time_taken[len(amount_time_taken)//2])/2)

    bins = [300+12.5*x for x in range(17)]
    plt.hist(amount_time_taken, bins)
    plt.savefig('histogram.png')
