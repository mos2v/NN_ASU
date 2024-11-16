import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

df = pd.read_csv('birds.csv')
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
a = df.isna().sum()
print(a[a>0])

mapping = {'A': -1, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(mapping)

encoder = OrdinalEncoder()
df['gender'] = encoder.fit_transform(df[['gender']])



A_y = df['bird category'][:50]
B_y = df['bird category'][50:100]
C_y = df['bird category'][100:]

X = df.drop(columns='bird category')
constant = 1e-5
X = np.log(X + constant)

A_x = X[:50]
B_x = X[50:100]
C_x = X[100:]

Y = pd.concat([A_y, B_y], ignore_index=True)
X = pd.concat([A_x, B_x], ignore_index=True)
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=42, stratify=Y)

# print(X_train.shape)
# print(X_test.shape)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

def adaline_predict(x, weights, bias = 0):
    predictions = []
    for index, row in x.iterrows():
        net_value = NetValue_calc(row.values, weights, bias)
        print(net_value)
        if net_value < 0:
            y_pred = -1
        else:
            y_pred = 1
        predictions.append(y_pred)
    return np.array(predictions)


def adaline_train(x, y, weights, epochs, lr, mse_threshold, bias = 0):
    for epoch in range(epochs):
        net_inputs = x.dot(weights) + bias
        errors = y - net_inputs
        weights_update = lr * x.T.dot(errors) / x.shape[0]
        bias_update = lr * errors.sum() / x.shape[0]

        weights += weights_update
        bias += bias_update

        mse = (errors**2).mean()
        # print(f'Epoch {epoch+1}, MSE: {mse}')
        if mse < mse_threshold:
            break

    return weights, bias


def perceptron_train(x, y, weights, epochs, lr, bias = 0):
    for i in range(epochs):
        for index, row in x.iterrows():
            vals = row.values
            net_value = NetValue_calc(vals, weights, bias)
            ypred = signum(net_value)
            error = y[index] - ypred

            if error == 0:
                continue

            weights = weights_calc(weights, lr, error, vals)

    return weights

def NetValue_calc(x, weights, bias=0):
    return np.dot(x, weights) + bias

def signum(v):
    if v > 0:
        return 1
    else:
        return -1

def weights_calc(weights, lr, error, x):
    return weights + (lr * error * x)


def perceptron_predict(x, weights, bias=0):
    predictions = []
    for index, row in x.iterrows():
        net_value = NetValue_calc(row.values, weights, bias)
        y_pred = signum(net_value)
        predictions.append(y_pred)
    return np.array(predictions)

weights = np.random.rand(X_train.shape[1]) * 0.01

epochs = 600
learning_rate = 0.01
# weights = perceptron_train(X_train, Y_train, weights, epochs, learning_rate)

# predictions = perceptron_predict(X_test, weights)


mse = 0.01
weights, bias = adaline_train(X_train, Y_train, weights, epochs, learning_rate, mse, 1)
predictions = adaline_predict(X_test, weights, bias)
print(predictions)

accuracy = np.mean(predictions == Y_test)
print("Accuracy on the test set:", accuracy)

print(classification_report(Y_test, predictions))

conf_matrix = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

def plot_decision_boundary(weights, bias=0, x_range=(0, 1)):

    slope = -weights[0] / weights[1]
    intercept = -bias / weights[1]


    x_vals = np.linspace(x_range[0], x_range[1], 100)


    y_vals = slope * x_vals + intercept

    return x_vals, y_vals

def plot_with_train_data(X_train, Y_train, weights, bias=0):

    x_vals, y_vals = plot_decision_boundary(weights, bias, x_range=(X_train.min().min(), X_train.max().max()))


    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, cmap='viridis', edgecolor='k', label='Training Data Points')

    plt.plot(x_vals, y_vals, 'r-', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary (Training Data)')
    plt.legend()
    plt.show()

# Plot the decision boundary with the training data
plot_with_train_data(X_train, Y_train, weights)
