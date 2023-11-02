import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def perceptron(input_data, weights, bias):
    activation = np.dot(input_data, weights) + bias
    output = sigmoid(activation)
    return output


inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
weights = np.array([(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
bias = 0
err = 100
result = []

for i in range(0, len(inputs), 1):
    output = perceptron(inputs[i], weights[i], bias)
    result.append(output)

print("Saída do perceptron:", result)
