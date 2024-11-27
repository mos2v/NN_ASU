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
            errors[-1] = activation[-1] - y  # For softmax with cross-entropy

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
            if l == len(weights) - 1:  # Output layer
                a = softmax(z)
            else:
                a = sigmoid(z) if activation_func else hyperbolic_tangent(z)
            activations.append(a)
        predicted_class = np.argmax(activations[-1])
        predictions.append(predicted_class)
    return predictions

def apply_model(hidden_layers, neurons, lr, epochs):
    df = pd.read_csv('birds.csv')
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
    encoder = OrdinalEncoder()
    df['gender'] = encoder.fit_transform(df[['gender']])

    Y = df['bird category']
    X = df.drop(columns='bird category')

    mapping = {'A': 0, 'B': 1, 'C': 2}
    Y = Y.map(mapping)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, shuffle=True, random_state=42, stratify=Y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(Y))
    Y_train_values = Y_train.reset_index(drop=True).values
    Y_test_values = Y_test.reset_index(drop=True).values
    Y_train_one_hot = np.eye(num_classes)[Y_train_values]
    Y_test_one_hot = np.eye(num_classes)[Y_test_values]

    weights, biases = MLP_train(
        X_train_scaled, Y_train_one_hot, hidden_layers, neurons, lr, epochs, activation_func=True, bias=True)
    predictions = MLP_predict(X_test_scaled, weights, biases, activation_func=True, bias=True)

    accuracy = np.mean(predictions == Y_test_values)
    print("Accuracy on the test set:", accuracy)
    print(classification_report(Y_test_values, predictions))
    conf_matrix = confusion_matrix(Y_test_values, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

# Example usage:
apply_model(hidden_layers=1, neurons=5, lr=0.05, epochs=1000)
