import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('birds.csv')


df['gender'] = df['gender'].fillna(df['gender'].mode()[0])


encoder = OrdinalEncoder()
df['gender'] = encoder.fit_transform(df[['gender']])

# Function definitions
def adaline_predict(x, weights, bias=0):
    net_inputs = np.dot(x, weights) + bias
    predictions = np.where(net_inputs < 0, -1, 1)
    return predictions

def adaline_train(x, y, weights, epochs, lr, mse_threshold, bias=0):
    bias_value = bias
    x_values = x.astype(np.float64)
    y_values = y.astype(np.float64)


    for epoch in range(epochs):
        net_inputs = np.dot(x_values, weights) + bias_value
        errors = y_values - net_inputs

        weights_update = lr * np.dot(x_values.T, errors)
        bias_update = lr * np.sum(errors)

        weights += weights_update
        bias_value += bias_update

        mse = np.mean(errors ** 2)
        # print(f"Epoch {epoch+1}, MSE: {mse}")

        if mse < mse_threshold:
            print("MSE threshold reached. Stopping training.")
            break

    return weights, bias_value

def perceptron_train(x, y, weights, epochs, lr, bias=0):
    bias_value = bias
    x_values = x
    y_values = y

    for epoch in range(epochs):
        for i in range(x_values.shape[0]):
            net_input = np.dot(x_values[i], weights) + bias_value
            y_pred = signum(net_input)
            error = y_values[i] - y_pred

            weights += lr * error * x_values[i]
            bias_value += lr * error

    return weights, bias_value

def perceptron_predict(x, weights, bias=0):
    net_inputs = np.dot(x, weights) + bias
    predictions = np.where(net_inputs < 0, -1, 1)
    return predictions

def signum(v):
    return np.where(v >= 0, 1, -1)

def plot_decision_boundary(weights, bias=0, x_range=(0, 1)):
    w1, w2 = weights[0], weights[1]
    slope = -w1 / w2
    intercept = -bias / w2

    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = slope * x_vals + intercept

    return x_vals, y_vals

def plot_with_train_data(X_train, Y_train, weights, bias=0):
    # If X_train is a NumPy array, convert it to a DataFrame
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2'])
    # If Y_train is a NumPy array, convert it to a Series
    if isinstance(Y_train, np.ndarray):
        Y_train = pd.Series(Y_train)

    # Get feature ranges
    x_min, x_max = X_train.iloc[:, 0].min(), X_train.iloc[:, 0].max()
    y_min, y_max = X_train.iloc[:, 1].min(), X_train.iloc[:, 1].max()

    x_vals, y_vals = plot_decision_boundary(weights, bias, x_range=(x_min, x_max))

    # Plot data points
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, cmap='viridis', edgecolor='k', label='Training Data Points')

    plt.plot(x_vals, y_vals, 'r-', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()

    # Show the plot without blocking
    plt.show(block=False)
    # Pause for 3 seconds
    plt.pause(6)
    # Close the plot window automatically
    plt.close()

def apply_model(feat1, feat2, class_values, lr, mse, bias, algo, epochs):

    df_filtered = df[df['bird category'].isin(class_values)]

    X = df_filtered[[feat1, feat2]]
    Y = df_filtered['bird category']

    class_map = {class_values[0]: -1, class_values[1]: 1}
    Y = Y.map(class_map)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=42, stratify=Y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize weights
    weights = np.random.randn(X_train_scaled.shape[1]) * 0.01

    # Convert Y_train and Y_test to NumPy arrays
    Y_train_values = Y_train.reset_index(drop=True).values
    Y_test_values = Y_test.reset_index(drop=True).values

    # Train the model
    if algo == 'Perceptron':
        weights, bias_value = perceptron_train(X_train_scaled, Y_train_values, weights, epochs=epochs, lr=lr, bias=bias)
        predictions = perceptron_predict(X_test_scaled, weights, bias_value)
    else:
        weights, bias_value = adaline_train(X_train_scaled, Y_train_values, weights, epochs=epochs, lr=lr, mse_threshold=mse, bias=bias)
        predictions = adaline_predict(X_test_scaled, weights, bias_value)

    # Evaluate the model
    accuracy = np.mean(predictions == Y_test_values)
    print("Accuracy on the test set:", accuracy)
    print(classification_report(Y_test_values, predictions))

    conf_matrix = confusion_matrix(Y_test_values, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    plot_with_train_data(X_train_scaled, Y_train_values, weights, bias_value)

def run_model():

    # Retrieve user inputs
    feat1 = feature1.get()
    feat2 = feature2.get()
    class_choice = selected_classes.get()
    lr = learning_rate.get()
    mse = mse_threshold.get()
    bias = 1 if use_bias.get() else 0
    algo = algorithm.get()
    epochs = num_epochs.get()

    # Validate inputs
    if not feat1 or not feat2:
        messagebox.showerror("Input Error", "Please select two features.")
        return
    if feat1 == feat2:
        messagebox.showerror("Input Error", "Please select two different features.")
        return
    if not class_choice:
        messagebox.showerror("Input Error", "Please select two classes.")
        return
    if epochs <= 0:
        messagebox.showerror("Input Error", "Number of epochs must be positive.")
        return

    apply_model(feat1, feat2, classes[class_choice], lr, mse, bias, algo, epochs)

# Initialize the main window
root = tk.Tk()
root.title("Bird Classification GUI")

# Get the list of features from the dataframe
feature_list = df.columns.tolist()
feature_list.remove('bird category')  # Remove the target variable

# Define classes
classes = {'A & B': ['A', 'B'], 'A & C': ['A', 'C'], 'B & C': ['B', 'C']}

# Variables to store user inputs
selected_features = []
selected_classes = tk.StringVar()
learning_rate = tk.DoubleVar(value=0.01)
mse_threshold = tk.DoubleVar(value=0.01)
use_bias = tk.BooleanVar()
algorithm = tk.StringVar(value='Perceptron')
num_epochs = tk.IntVar(value=600)  # Default number of epochs

# Create GUI components
# Feature Selection
feature_label = tk.Label(root, text="Select Two Features:")
feature_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')

feature1 = ttk.Combobox(root, values=feature_list)
feature1.grid(row=1, column=0, padx=10, pady=5)
feature2 = ttk.Combobox(root, values=feature_list)
feature2.grid(row=1, column=1, padx=10, pady=5)

# Class Selection
class_label = tk.Label(root, text="Select Two Classes:")
class_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')

class_options = ttk.Combobox(root, values=list(classes.keys()), textvariable=selected_classes)
class_options.grid(row=3, column=0, padx=10, pady=5)

# Learning Rate Input
lr_label = tk.Label(root, text="Learning Rate:")
lr_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')

lr_entry = tk.Entry(root, textvariable=learning_rate)
lr_entry.grid(row=5, column=0, padx=10, pady=5)

# Epochs Input
epochs_label = tk.Label(root, text="Number of Epochs:")
epochs_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')

epochs_entry = tk.Entry(root, textvariable=num_epochs)
epochs_entry.grid(row=7, column=0, padx=10, pady=5)

# MSE Threshold Input
mse_label = tk.Label(root, text="MSE Threshold:")
mse_label.grid(row=8, column=0, padx=10, pady=5, sticky='w')

mse_entry = tk.Entry(root, textvariable=mse_threshold)
mse_entry.grid(row=9, column=0, padx=10, pady=5)

# Bias Checkbox
bias_checkbox = tk.Checkbutton(root, text="Add Bias", variable=use_bias)
bias_checkbox.grid(row=10, column=0, padx=10, pady=5, sticky='w')

# Algorithm Selection
alg_label = tk.Label(root, text="Choose Algorithm:")
alg_label.grid(row=11, column=0, padx=10, pady=5, sticky='w')

alg_options = [('Perceptron', 'Perceptron'), ('Adaline', 'Adaline')]
for idx, (text, mode) in enumerate(alg_options):
    alg_radio = tk.Radiobutton(root, text=text, variable=algorithm, value=mode)
    alg_radio.grid(row=12+idx, column=0, padx=10, pady=5, sticky='w')

# Run Button
run_button = tk.Button(root, text="Run", command=run_model)
run_button.grid(row=14, column=0, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
