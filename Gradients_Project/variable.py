import math
import numpy as np

class Variable():
    '''
    A class to represent an algebraic variable, for the purpose of evaluating expressions and calculating gradients numerically.

    Attributes
    ----------
    name : str
        name of the variable
    evaluate : function
        a function which takes in a dictionary of values for each variable name and returns the value of the variable
    gradient : function
        a function which takes in a dictionary of values for each variable name and returns the gradient of the variable
    '''

    def __init__(self, name=None, evaluate=None, gradient=None):
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate

        if name != None:
            self.name = name # its key in the evaluation dictionary

        if gradient == None:
            self.gradient = lambda values: np.array(list(map((lambda x: int(x == self.name)), sorted(list(values.keys()))))) # generates the gradient of a single independent variable (array of all 0s and a single 1)
        else:
            self.gradient = gradient
    
    def __add__(self, other):
        if type(other) in (int, float): # add a float to a variable
            return Variable(evaluate = lambda values: self.evaluate(values) + other, gradient = lambda values: self.gradient(values))
        
        # add a variable to a variable
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values), gradient = lambda values: self.gradient(values) + other.gradient(values))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if type(other) in (int, float): # multiply a variable by a float
            return Variable(evaluate = lambda values: self.evaluate(values) * other, gradient = lambda values: other * self.gradient(values))

        # multiply a variable by a variable
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values), gradient = lambda values: other.evaluate(values) * self.gradient(values) + other.gradient(values) * self.evaluate(values))
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if type(other) in (int, float): # raise a variable to a float power
            return Variable(evaluate = lambda values: self.evaluate(values) ** other, gradient = lambda values: other * (self.evaluate(values)) ** (other - 1) * self.gradient(values))
        
        # raise a variable to a variable power, leveraging the power of the exp and log functions defined later
        return Variable.exp(other * Variable.log(self))

    def __rpow__(self, other):
        if type(other) in (int, float): # raising a float to a variable power
            return Variable(evaluate = lambda values: other ** self.evaluate(values), gradient = lambda values: (other ** self.evaluate(values)) * math.log(other) * self.gradient(values))

    # we get all these functions "for free" now
    def __sub__(self, other):
        return self + (other * -1)
    
    def __rsub__(self, other):
        return -1 * (self - other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __call__(self, values):
        return self.evaluate(values)

    def safe_eval(self, values, max=None, min=None):
        '''Evaluates a variable with a minimum (exclusive) and maximum (inclusive),
        necessary to check the domain of variables in log.'''
        attempt = self.evaluate(values)
        if max != None:
            if attempt > max:
                return max
        if min != None:
            if attempt <= min:
                return np.nextafter(min,min+1)

        return attempt

    # method to avoid division by zero errors in later gradient computation 
    @staticmethod
    def nonzero(num):
        if num == 0:
            return math.e**(-20)
        return num

    # static methods for exp and log
    @staticmethod
    def exp(var):
        if type(var) in (int, float): # raising e to the power of a float
            return Variable(evaluate = lambda values: math.e ** var, gradient = lambda values: np.zeros(len(values)))

        # raising e to the power of a variable
        return Variable(evaluate = lambda values: math.e ** var.evaluate(values), gradient = lambda values: (math.e ** var.evaluate(values)) * var.gradient(values))

    @staticmethod
    def log(var):
        if type(var) in (int, float): # taking the log of a float
            return Variable(evaluate = lambda values: math.log(var) if var > 0 else math.log(np.nextafter(var,var+1)), gradient = lambda values: np.zeros(len(values)))

        # taking the log of a variable
        return Variable(evaluate = lambda values: math.log(var.safe_eval(values, min=0)), gradient = lambda values: var.gradient(values) / Variable.nonzero(var.evaluate(values)))