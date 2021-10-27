import math
import numpy as np
import colorama
from colorama import Fore
from variable import Variable

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, dataset, labels, learning_rate, total_times):
        n, d = np.shape(dataset)
        mvars = []
        for i in range(d):
            mvars.append(Variable(name=f'm_{i}'))
        bvar = Variable(name='b')

        y_hats = []
        for i in range(n):
            y_hats.append(1/(1+Variable.exp(sum(np.array(mvars)*dataset[i]) + bvar)))

        lossvar = (-1)*sum([labels[i]*Variable.log(y_hats[i]) + (1-labels[i])*Variable.log(1-y_hats[i]) for i in range(n)])

        self.ms = np.random.rand(d)-0.5
        self.b = np.random.random_sample()-0.5

        self.best_ms = self.ms
        self.best_b = self.b
        dict_values = {'b':self.b}
        for i in range(d):
            dict_values.update({f'm_{i}':self.ms[i]})
        self.best_loss = lossvar.evaluate(dict_values)

        num_times = 0

        while True:
            dict_values = {'b':self.b}
            for i in range(d):
                dict_values.update({f'm_{i}':self.ms[i]})

            step = lossvar.gradient(dict_values)

            self.b -= learning_rate * step[0]
            self.ms -= learning_rate * step[1:]

            loss = lossvar.evaluate(dict_values)
            
            if (loss < self.best_loss):
                self.best_ms = self.ms
                self.best_b = self.b
                self.best_loss = loss
                self.best_found_at = num_times

            if (num_times % 1000 == 0):
                print(Fore.RED+f'Loss: {loss}', Fore.RESET+f'Step: {step}',f'Current b: {self.b}',f'Current ms: {self.ms}',sep='\n')

            if (num_times >= total_times):
                print('------------------------------------')
                print(Fore.BLUE + f'Training took {num_times} iteration(s).')
                print(f'Final Loss: {loss}',f'Final b: {self.b}',f'Final ms: {self.ms}',sep='\n')
                print(f'Best Loss: {self.best_loss}',f'Best b: {self.best_b}',f'Best ms: {self.best_ms}',f'Found on iteration {self.best_found_at}'+Fore.RESET,sep='\n')
                break

            num_times += 1

        # print(list(map(lambda x:x.name,self.ms)))

    def predict(self, dataset):
        n, _ = np.shape(dataset)
        predictions = []

        for i in range(n):
            predictions.append(1/(1+math.exp(sum(np.array(self.ms)*dataset[i]) + self.b)))

        return predictions    