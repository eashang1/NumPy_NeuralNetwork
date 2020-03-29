#based on video by Polycode
import numpy as np

#normalizing function (returns a probability)
def sigmoid(x):
    return 1/(1+np.exp(-x))

#calculates weighted derivative 
def sigmoidDerivative(x):
    return x*(1-x)

'''
Training data:
0 0 1   0
1 1 1   1
1 0 1   1
0 1 1   0

Note that the weight of the first input column should increase, as it is the
best predictor of the output
'''
trainingInputs = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

#.T transposes matrix
trainingOutputs = np.array([[0,1,1,0]]).T

#weights are initilaized at random from -1 to 1  in a 3 by 1 matrix
np.random.seed(1)
weights = 2*np.random.random((3,1))-1

print('Random starting weights: ')
print(weights)

#trains the neural network
for iteration in range(100000):

    inputLayer = trainingInputs

    #output is normalized weighted sum (sigmoid function of dot product )
    outputs = sigmoid(np.dot(inputLayer, weights))

    '''
    Backpropagation (adjusts weights)
    1) error is calculated as difference between expected and actual outputs
    
    2) adjustment matrix is the product of the error with sigmoid derivative
    
        -error factor creates greater change for greater error
        
        -derivative graph levels out at either end, so derivative factor creates
         less change if the network is more confident in a certain relationship
         
    3) adjustment is only multiplied to a weight if the corresponding input is 1
    '''
    error = trainingOutputs - outputs
    adjustments = error*(sigmoidDerivative(outputs))
    weights += np.dot(inputLayer.T, adjustments)

print('Weights after training: ')
print(weights)

print('Outputs after training: ')
print(outputs)
