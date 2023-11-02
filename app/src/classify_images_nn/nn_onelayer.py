import numpy as np


def step_function(x):
    return 1 if x >= 1 else 0


def perceptron(input_data, weights, bias):
    activation = np.dot(input_data, weights) + bias
    output = step_function(activation)
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
