from cmath import exp

import numpy as np
import scipy.special

def activation_reLU(x):
    return 0 if x < 0 else x

class neuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate, activationFunc):
        self.inNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outNodes = outputNodes
        self.learningRate = learningRate
        if (activationFunc == 'simoid'):
            self.activation_func = scipy.special.expit
        if (activationFunc == 'simoid'):
            self.activation_func = activation_reLU
        #self.activation_func = activation_logit
        self.wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inNodes))
        self.who = np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hiddenNodes))
        print('Инициализация выполнена')
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndim=2).T
        targets = np.array(targets_list, ndim=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learningRate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.learningRate * np.dot((hidden_errors * hidden_outputs* (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        #из list в матрицу numpy
        inputs = np.array(inputs_list, ndmin=2).T

        # Wih * inputs
        # входные сигналы на веса скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        # Who * hidden_outputs
        # сигналы скрытого слоя на веса выходного слоя
        final_inputs = np.dot(self.who, hidden_inputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

nn = neuralNetwork(3, 3, 3, 0.5)

inp = list([1, 1.1, 1.2])
res = nn.query(inp)
print(res)

