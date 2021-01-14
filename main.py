import numpy as np
import re
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons, bias=True):
    hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
    # hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
    # output_bias = np.random.uniform(size=(1, outputLayerNeurons))

    if bias:
        hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
        output_bias = np.random.uniform(size=(1, outputLayerNeurons))
        return hidden_weights, hidden_bias, output_weights, output_bias
    else:
        return hidden_weights, None, output_weights, None


def forward_propagation(inputs, hidden_weights, hidden_bias, output_weights, output_bias):
    hidden_layer_activation = np.matmul(inputs, hidden_weights)
    if hidden_bias is not None:
        hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.matmul(hidden_layer_output, output_weights)
    if output_bias is not None:
        output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    return hidden_layer_output, predicted_output


def backward_propagation(expected_output, predicted_output, output_weights, hidden_layer_output):
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = np.matmul(d_predicted_output, output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    return d_hidden_layer, d_predicted_output


def update(output_weights, hidden_weights, hidden_layer_output, d_predicted_output,
           d_hidden_layer, inputs, lr, output_bias, hidden_bias):
    output_weights += np.matmul(hidden_layer_output.T, d_predicted_output) * lr

    # output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += np.matmul(inputs.T, d_hidden_layer) * lr
    # hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
    if output_bias is not None and hidden_bias is not None:
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


def train(inputs, expected_output, hidden_weights, hidden_bias, output_weights, output_bias, epochs, lr):
    for _ in range(epochs):
        # Forward Propagation
        hidden_layer_output, predicted_output = forward_propagation(inputs, hidden_weights, hidden_bias,
                                                                    output_weights, output_bias)

        # Backpropagation
        d_hidden_layer, d_predicted_output = backward_propagation(expected_output, predicted_output,
                                                                  output_weights, hidden_layer_output)

        # Updating Weights and Biases
        update(output_weights, hidden_weights, hidden_layer_output, d_predicted_output,
           d_hidden_layer, inputs, lr, output_bias, hidden_bias)

    print("Final hidden weights: ", end='')
    print(*hidden_weights)
    if hidden_bias is not None:
        print("Final hidden bias: ", end='')
        print(*hidden_bias)
    print("Final output weights: ", end='')
    print(*output_weights)
    if output_bias is not None:
        print("Final output bias: ", end='')
        print(*output_bias)
    predicted_output = np.around(predicted_output, decimals=0)
    return predicted_output


if __name__ == "__main__":
    start_time = time.time()
    print("Zadanie 1 (bez biasu)")
    for i in range(1, 4):
        print(f"Ilość neuronów warstwy ukrytej: {i}")
        inputs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(4, i, 4, False)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*predicted_output)
        print()

    print("Zadanie 1 (z biasem)")
    for i in range(1, 4):
        print(f"Ilość neuronów warstwy ukrytej: {i}")
        inputs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(4, i, 4)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*predicted_output)
        print()

    print("Zadanie 2 (bez biasu)")
    for i in range(1, 4):
        print(f"Ilość neuronów warstwy ukrytej: {i}")
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected_output = np.array([[0], [1], [1], [0]])
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(2, 3, 1, False)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*np.around(predicted_output, decimals=0))


    print("Zadanie 2 (z biasem)")
    for i in range(1, 4):
        print(f"Ilość neuronów warstwy ukrytej: {i}")
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected_output = np.array([[0], [1], [1], [0]])
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(2, 3, 1)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*np.around(predicted_output, decimals=0))

    print(f"Zajęło to dokładnie: {time.time() - start_time}")


# Input datasets
# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# expected_output = np.array([[0], [1], [1], [0]])
# inputs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#
# epochs = 10000
# lr = 0.1
# inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 4, 3, 4
# # inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 3, 1
#
# # Random weights and bias initialization
# hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
# hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
# output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
# output_bias = np.random.uniform(size=(1, outputLayerNeurons))
#
# print("Initial hidden weights: ", end='')
# print(*hidden_weights)
# print("Initial hidden biases: ", end='')
# print(*hidden_bias)
# print("Initial output weights: ", end='')
# print(*output_weights)
# print("Initial output biases: ", end='')
# print(*output_bias)
#
# # Training algorithm
# for _ in range(epochs):
#     # Forward Propagation
#     hidden_layer_activation = np.matmul(inputs, hidden_weights)
#     hidden_layer_activation += hidden_bias
#     hidden_layer_output = sigmoid(hidden_layer_activation)
#
#     output_layer_activation = np.matmul(hidden_layer_output, output_weights)
#     output_layer_activation += output_bias
#     predicted_output = sigmoid(output_layer_activation)
#
#     # Backpropagation
#     error = expected_output - predicted_output
#     d_predicted_output = error * sigmoid_derivative(predicted_output)
#
#     error_hidden_layer = np.matmul(d_predicted_output, output_weights.T)
#     d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
#
#     # Updating Weights and Biases
#     output_weights += np.matmul(hidden_layer_output.T, d_predicted_output) * lr
#
#     output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
#     hidden_weights += np.matmul(inputs.T, d_hidden_layer) * lr
#     hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
#
# print("Final hidden weights: ", end='')
# print(*hidden_weights)
# print("Final hidden bias: ", end='')
# print(*hidden_bias)
# print("Final output weights: ", end='')
# print(*output_weights)
# print("Final output bias: ", end='')
# print(*output_bias)
#
# predicted_output = np.around(predicted_output, decimals=0)
#
# print("\nOutput from neural network after 10,000 epochs: ", end='')
# print(*predicted_output)
# print(re.sub("[^0-9\s]", "", ' '.join([str(elem) for elem in predicted_output])))
