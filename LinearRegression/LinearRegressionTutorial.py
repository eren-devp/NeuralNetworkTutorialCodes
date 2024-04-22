import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style
import sklearn
from sklearn import linear_model

# Note: Regression is like: y = mx + b

data = pd.read_csv("student-mat.csv", sep=";")  # Reading the dataset.
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Datas that we need for predict.
predict = "G3"  # We will try to predict third grade.

x = np.array(
    data.drop([predict], 1))  # Dropping the "predict" data. We can't guess the thing that we know already right?

y = np.array(data[predict])  # We will compare the predict with the real results so we can train the AI.
test_size = 0.1  # This is the percentage of the data that we will be using for training.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
print(data.drop([predict], 1))
""" The training and finding the best regression's process.
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)

    linear = linear_model.LinearRegression()  # We will use the linear regression because these variables are directly influences the "predict".

    linear.fit(x_train, y_train)  # Fitting the data to find the best function.
    acc = linear.score(x_test, y_test)  # Getting the accuracy of the fitted data.
    print("Acc: ", acc)

    if acc > best:
        best = acc
        with open("student-model.pickle", "wb") as f:
            pickle.dump(linear, f)  # Saving the best model.
"""
pickle_in = open("student-model.pickle", "rb")
linear = pickle.load(pickle_in)  # Loading the best model.

print("Co: ", linear.coef_)  # Coefficients of the data columns.
print("Intercept: ", linear.intercept_)  # The constant (b) value of our regression.

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")  # These things are for be able to see our regression with graphics.
p = "absences"  # This data will be the x label.
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(f"Final Grade ({predict})")
pyplot.show()
