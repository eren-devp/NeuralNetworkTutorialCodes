import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# https://www.youtube.com/watch?v=vwLT6bZrHEE&list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr&index=6

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()  # Encoder for the non-numerical data into numerical data.
buying = le.fit_transform(list(data["buying"]))  # The transformations process for the "buying" data.
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # Converting these datas into one big list.
y = list(cls)

test_size = 0.1
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)

k_weight = 9
model = KNeighborsClassifier(n_neighbors=k_weight)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

prediction = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", names[prediction[x]], " Data: ", x_test[x], " Actual: ", names[y_test[x]])
    neighbors = model.kneighbors([x_test[x]], k_weight, True)  # RTFM!
    print("Neighbors: ", neighbors)
