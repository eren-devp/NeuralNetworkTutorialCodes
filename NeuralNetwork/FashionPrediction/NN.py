import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist  # Importing our fashion data.

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

train_images = train_images / 255.0
test_images = test_images / 255.0

# https://www.youtube.com/watch?v=cvNtZqphr6A&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=3
hidden_neuron_count = 128
output_neuron_count = len(class_names)
img_sizes = (28, 28)
epoch_count = 3

model = keras.Sequential([
    keras.layers.Flatten(input_shape=img_sizes),
    keras.layers.Dense(hidden_neuron_count, activation="relu"),  # Dense means fully-connected layers.
    keras.layers.Dense(output_neuron_count, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Just heard adam is a standard as a optimizer lol.
model.fit(train_images, train_labels, epochs=epoch_count)  # Epoch is giving the same inputs with random orders.

"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc: ", test_acc)
"""

prediction = model.predict(test_images[7])  # Getting as much output as output neuron count.
# print(class_names[np.argmax(prediction[0])])  # Getting the output that has highest probability.

"""
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
"""
