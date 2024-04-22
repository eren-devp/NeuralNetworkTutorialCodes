import numpy as np
from sklearn import svm
from random import random

def foo(x1, x2, x3):
    return x1 ** x2 - x2 * 12 + x3 * 4 - 31

train_data_size = 10000
test_data_size = 50
train_results = []
train_inputs = []
for _ in range(train_data_size):  # Creating the train data-set.
    x1 = random()
    x2 = random()
    x3 = random()
    train_inputs.append([x1, x2, x3])
    train_results.append(foo(x1, x2, x3))

test_inputs = []
test_results = []
for _ in range(test_data_size):  # Creating the test data-set.
    x1 = random()
    x2 = random()
    x3 = random()
    test_inputs.append([x1, x2, x3])
    test_results.append(foo(x1, x2, x3))

train_inputs = np.array(train_inputs)  # Converting them into numpy array.
train_results = np.array(train_results)
test_inputs = np.array(test_inputs)
test_results = np.array(test_results)

ep = 0.1
clf = svm.SVR(kernel="sigmoid", shrinking=False)
clf.fit(test_inputs, test_results)

x1 = float(input("X1: "))
x2 = float(input("X2: "))
x3 = float(input("X3: "))

inp_array = np.array([[x1, x2, x3]])
prediction = round(clf.predict(inp_array)[0] * 100) / 100
answer = round(foo(x1, x2, x3) * 100) / 100

print(f"Prediction: {prediction} Result: {answer} Distance:"
      f"{round(abs(prediction - answer) * 100) / 100}\n")

#y_prediction = clf.predict(test_inputs)
#acc = metrics.accuracy_score(test_results, y_prediction)
#print(acc)