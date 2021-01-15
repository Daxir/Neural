import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons, bias=True):
    hidden_weights = np.random.default_rng().random(size=(inputLayerNeurons, hiddenLayerNeurons))
    output_weights = np.random.default_rng().random(size=(hiddenLayerNeurons, outputLayerNeurons))

    if bias:
        hidden_bias = np.random.default_rng().random(size=(1, hiddenLayerNeurons))
        output_bias = np.random.default_rng().random(size=(1, outputLayerNeurons))
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

    hidden_weights += np.matmul(inputs.T, d_hidden_layer) * lr
    if output_bias is not None and hidden_bias is not None:
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


def train(inputs, expected_output, hidden_weights, hidden_bias, output_weights, output_bias, epochs, lr):
    for _ in range(epochs):
        # propagacja w przód
        hidden_layer_output, predicted_output = forward_propagation(inputs, hidden_weights, hidden_bias,
                                                                    output_weights, output_bias)

        # propagacja wsteczna
        d_hidden_layer, d_predicted_output = backward_propagation(expected_output, predicted_output,
                                                                  output_weights, hidden_layer_output)

        # aktualizacja wag i biasu
        update(output_weights, hidden_weights, hidden_layer_output, d_predicted_output,
           d_hidden_layer, inputs, lr, output_bias, hidden_bias)

    print("Wagi końcowe warstwy ukrytej: ", end='')
    print(*hidden_weights)
    if hidden_bias is not None:
        print("Końcowy bias warstwy ukrytej: ", end='')
        print(*hidden_bias)
    print("Wagi końcowe wyjścia: ", end='')
    print(*output_weights)
    if output_bias is not None:
        print("Końcowy bias wyjścia: ", end='')
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
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(2, i, 1, False)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*predicted_output)


    print("Zadanie 2 (z biasem)")
    for i in range(1, 4):
        print(f"Ilość neuronów warstwy ukrytej: {i}")
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected_output = np.array([[0], [1], [1], [0]])
        hidden_weights, hidden_bias, output_weights, output_bias = initialize(2, i, 1)
        predicted_output = train(inputs, expected_output, hidden_weights,
                                 hidden_bias, output_weights, output_bias, 10000, 0.1)
        print(*predicted_output)

    print(f"Zajęło to dokładnie: {time.time() - start_time}")

