#based on video by Polycode
import numpy as np
#see perceptron_explanation for more details on functionality
class NeuralNetwork():

    def __init__(self):
        self.weights = 2*np.random.random((3,1))-1

    #normalization function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #calculates gradient of sigmoid function
    def sigmoid_derivative(self, x):
        return x*(1-x)

    #carries out learning
    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range (training_iterations):

            output = self.think(training_inputs)

            #backpropagation
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error*self.sigmoid_derivative(output))
            self.weights += adjustments

    #forward propagation
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output

#main method
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print('Random weights: ')
    print(neural_network.weights)

    training_inputs = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_inputs, training_outputs, 100000);
    print('Weights after training: ')
    print(neural_network.weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A,B,C])))
