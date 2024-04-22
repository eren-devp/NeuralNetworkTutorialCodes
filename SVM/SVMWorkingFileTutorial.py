import sklearn
from sklearn import datasets  # Datasets that built-in in sklearn.
from sklearn import svm

# https://www.youtube.com/watch?v=JHxyrMgOUWI&list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr&index=9
# Kernel is like a dimension upgrade function that uses x1 ** 2 + x2 ** 2 = x3 to calculate upgraded dimension.
# Soft margin is allowing a few points to be incorrect for getting a better hyper-plane.

cancer_dataset = datasets.load_breast_cancer()

x = cancer_dataset.data
y = cancer_dataset.target

test_size = 0.2
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)

classes = ["malignant", "benign"]

soft_margin = 2
clf = svm.SVR(kernel="linear", C=soft_margin)  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
clf.fit(x_train, y_train)

#y_prediction = clf.predict(x_test)
#acc = metrics.accuracy_score(y_test, y_prediction)  # Calculation the accuracy.
#print(acc)

for x in range(len(x_test)):
    print("Prediction: ", clf.predict(x_test)[x], " Value: ", y_test[x])
