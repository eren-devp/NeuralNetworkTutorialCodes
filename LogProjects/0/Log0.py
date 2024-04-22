import numpy as np
import pickle
from sklearn import linear_model
from random import random


def foo(x1, x2, x3, x4):
    return x1 * 45 - x2 * 31 + x3 * 4 + x4 * 2 - 31


"""
train_data_size = 5
test_data_size = 50
train_results = []
train_inputs = []
for _ in range(train_data_size):  # Creating the train data-set.
    x1 = random()
    x2 = random()
    x3 = random()
    x4 = random()
    train_inputs.append([x1, x2, x3, x4])
    train_results.append(foo(x1, x2, x3, x4))

test_inputs = []
test_results = []
for _ in range(test_data_size):  # Creating the test data-set.
    x1 = random()
    x2 = random()
    x3 = random()
    x4 = random()
    test_inputs.append([x1, x2, x3, x4])
    test_results.append(foo(x1, x2, x3, x4))

train_inputs = np.array(train_inputs)  # Converting them into numpy array.
train_results = np.array(train_results)
test_inputs = np.array(test_inputs)
test_results = np.array(test_results)

best = 0
for _ in range(1):
    linear = linear_model.LinearRegression()

    linear.fit(train_inputs, train_results)
    acc = linear.score(test_inputs, test_results)
    print("Acc: ", acc)

    if acc > best:
        best = acc
        with open("Log0.pickle", "wb") as f:
            pickle.dump(linear, f)
#"""

pickle_in = open("Log0.pickle", "rb")
linear = pickle.load(pickle_in)

print("=====-> Linear Regression Example <-====")

run_time = 0
while True:
    run_time += 1
    try:
        print(f"Calculation {run_time}")
        x1 = float(input("X1: "))
        x2 = float(input("X2: "))
        x3 = float(input("X3: "))
        x4 = float(input("X4: "))

        inp_array = np.array([[x1, x2, x3, x4]])
        prediction = round(linear.predict(inp_array)[0] * 100) / 100
        answer = round(foo(x1, x2, x3, x4) * 100) / 100

        print("Co: ", linear.coef_)
        print("Intercept: ", linear.intercept_)
        print(f"Prediction: {prediction} Result: {answer} Distance:"
              f"{round(abs(prediction - answer) * 100) / 100}\n")

    except:
        print("Please enter valid numbers.\n")
