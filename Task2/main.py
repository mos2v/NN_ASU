import numpy as np
from numpy.ma.core import append
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


def hyperbolic_tangent(v):
    return (np.exp(v) - np.exp(-v)) / (np.exp(v) + np.exp(-v))

def tanh_derivative(v):
    f = hyperbolic_tangent(v)
    return 1 - f ** 2

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def sigmoid_derivative(v):
    f = sigmoid(v)
    return f * (1 - f)


def random_matrix(rows, cols):
    limit = np.sqrt(6 / (rows + cols))
    return np.random.uniform(-limit, limit, (rows, cols))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

# def random_matrix(rows, cols):
#     return np.random.rand(rows, cols) * 0.001

def random_vector(length):
    return np.random.rand(length) * 0.01

def MLP_train(X, Y, hidden_layers, neurons, learning_rate, epochs, activation_func, bias=False):
    input_size = X.shape[1]
    output_size = Y.shape[1]
    layers = [input_size] + [neurons] * hidden_layers + [output_size]
    weights = []
    biases = []
    for l in range(len(layers) - 1):
        weights.append(random_matrix(layers[l + 1], layers[l]))
        if bias:
            biases.append(random_vector(layers[l + 1]))

    for epoch in range(epochs):
        total_error = 0
        for x, y in zip(X, Y):
            activation = [x]
            zs = []

            # Forward pass
            for l in range(len(layers) - 1):
                z = np.dot(weights[l], activation[l])
                if bias:
                    z += biases[l]
                zs.append(z)

                if l == len(layers) - 2:  # Output layer
                    a = softmax(z)
                else:
                    a = sigmoid(z) if activation_func else hyperbolic_tangent(z)
                activation.append(a)

            # Compute error at the output layer
            errors = [None] * (len(layers) - 1)
            errors[-1] = activation[-1] - y

            # Cross-entropy loss
            total_error += -np.sum(y * np.log(activation[-1] + 1e-15))

            for l in reversed(range(len(layers) - 2)):
                derivative = sigmoid_derivative(zs[l]) if activation_func else tanh_derivative(zs[l])
                errors[l] = np.dot(weights[l + 1].T, errors[l + 1]) * derivative


            # Update weights and biases
            for l in range(len(layers) - 1):
                weights[l] -= learning_rate * np.outer(errors[l], activation[l])
                if bias:
                    biases[l] -= learning_rate * errors[l]

        average_error = total_error / len(X)
        print(f"Epoch {epoch + 1}, Average Cross-Entropy Loss: {average_error:.6f}")
    return weights, biases

def MLP_predict(X, weights, biases, activation_func, bias=False):
    predictions = []
    for x in X:
        activations = [x]
        for l in range(len(weights)):
            z = np.dot(weights[l], activations[l])
            if bias:
                z += biases[l]
            if l == len(weights) - 1:
                a = softmax(z)
            else:
                a = sigmoid(z) if activation_func else hyperbolic_tangent(z)
            activations.append(a)
        predicted_class = np.argmax(activations[-1])
        predictions.append(predicted_class)
    return predictions

def apply_model(hidden_layers, neurons, lr, epochs, activation, use_bias):
    df = pd.read_csv('birds.csv')
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
    encoder = OrdinalEncoder()
    df['gender'] = encoder.fit_transform(df[['gender']])

    Y = df['bird category']
    X = df.drop(columns='bird category')

    mapping = {'A': 0, 'B': 1, 'C': 2}
    Y = Y.map(mapping)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=42, stratify=Y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(Y))
    Y_train_values = Y_train.reset_index(drop=True).values
    Y_test_values = Y_test.reset_index(drop=True).values
    Y_train_one_hot = np.eye(num_classes)[Y_train_values]
    Y_test_one_hot = np.eye(num_classes)[Y_test_values]

    weights, biases = MLP_train(X_train_scaled, Y_train_one_hot, hidden_layers, neurons, lr, epochs, activation_func=True, bias=True)
    predictions = MLP_predict(X_test_scaled, weights, biases, activation_func=True, bias=True)

    accuracy = np.mean(predictions == Y_test_values)
    print("Accuracy on the test set:", accuracy)
    print(classification_report(Y_test_values, predictions))
    conf_matrix = confusion_matrix(Y_test_values, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)


def run_model():

    hiddenL = hidden_layers.get()
    neurons_num = neurons.get()
    lr = learning_rate.get()
    epochs = num_epochs.get()
    bias = True if use_bias.get() else False
    activ = True if activation.get() == 'Sigmoid' else False

    if hiddenL <= 0:
        messagebox.showerror("Input Error", "Number of hidden layers must be positive.")
        return

    if neurons_num <= 0:
        messagebox.showerror("Input Error", "Number of neurons must be positive.")
        return

    if epochs <= 0:
        messagebox.showerror("Input Error", "Number of epochs must be positive.")
        return

    apply_model(hiddenL, neurons_num, lr, epochs, activ, use_bias)

root = tk.Tk()
root.title("Bird Classification GUI")

# Variables

hidden_layers = tk.IntVar()
neurons = tk.IntVar()
learning_rate = tk.DoubleVar(value=0.01)
num_epochs = tk.IntVar(value=600)
use_bias = tk.BooleanVar()
activation = tk.StringVar()

hidden_layers_label = tk.Label(root, text="Hidden Layers:")
hidden_layers_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')

hidden_layer_entry = tk.Entry(root, textvariable=hidden_layers)
hidden_layer_entry.grid(row=3, column=0, padx=10, pady=5)

neurons_label = tk.Label(root, text="Number of Neurons:")
neurons_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')

neurons_entry = tk.Entry(root, textvariable=neurons)
neurons_entry.grid(row=5, column=0, padx=10, pady=5)

# Learning Rate Input
lr_label = tk.Label(root, text="Learning Rate:")
lr_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')

lr_entry = tk.Entry(root, textvariable=learning_rate)
lr_entry.grid(row=7, column=0, padx=10, pady=5)

# Epochs Input
epochs_label = tk.Label(root, text="Number of Epochs:")
epochs_label.grid(row=8, column=0, padx=10, pady=5, sticky='w')

epochs_entry = tk.Entry(root, textvariable=num_epochs)
epochs_entry.grid(row=9, column=0, padx=10, pady=5)

# Bias Checkbox
bias_checkbox = tk.Checkbutton(root, text="Add Bias", variable=use_bias)
bias_checkbox.grid(row=11, column=0, padx=10, pady=5, sticky='w')

# Algorithm Selection
activation_label = tk.Label(root, text="Choose Activation Function:")
activation_label.grid(row=12, column=0, padx=10, pady=5, sticky='w')

alg_options = [('Sigmoid', 'Sigmoid'), ('Hyperbolic Tangent', 'Hyperbolic Tangent')]
for idx, (text, mode) in enumerate(alg_options):
    alg_radio = tk.Radiobutton(root, text=text, variable=activation, value=mode)
    alg_radio.grid(row=13, column=idx, padx=10, pady=5, sticky='w')


run_button = tk.Button(root, text="Run", command=run_model)
run_button.grid(row=14, column=0, padx=10, pady=10)

root.mainloop()
