import numpy as np
import re

# np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# expected_output = np.array([[0], [1], [1], [0]])
inputs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 4, 3, 4
# inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 3, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)

# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.matmul(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.matmul(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = np.matmul(d_predicted_output, output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += np.matmul(hidden_layer_output.T, d_predicted_output) * lr

    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += np.matmul(inputs.T, d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)

predicted_output = np.around(predicted_output, decimals=0)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
# print(re.sub("[^0-9\s]", "", ' '.join([str(elem) for elem in predicted_output])))
