import tensorflow as tf
import numpy as np
from tensorflow import keras

data = keras.datasets.imdb
word_vector_count = 88000

# (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=word_vector_count)  # It's like limiting the word count.
# print(train_data[0])  # The words in the data are replaced by integers to make it easier to work with.

word_index = data.get_word_index()  # Gives the tuple of strings and integers.
word_index = {k: (v+3) for k, v in word_index.items()}  # First 4 keys are special chars so we won't be taking them.
word_index["<PAD>"] = 0  # Means padding. We will set all the data to a fixed char size. So with this "0" is going to be the padding.
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # Means unknown.
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # Swapping keys by values so we can replace(decode) the integer input to string input by using this dict.

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])  # If we can't find a value that caters the i we will send "?" as an output.

def encode_review(text):
    encoded = [0]

    for word in text:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)

    return encoded

# print(decode_review(test_data[0]))
# print(len(test_data[0]), len(test_data[1]))  # To see the datas have different length.
"""
# MODEL TRAINING
max_len_of_reviews = 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding="post", maxlen=max_len_of_reviews)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                       padding="post", maxlen=max_len_of_reviews)

model = keras.Sequential()
# https://www.youtube.com/watch?v=qpb_39IjZA0&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=6
model.add(keras.layers.Embedding(word_vector_count, 16))  # Adding layer to model.
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # We use sigmoid because there will be negative(-) reviews.

model.summary()
# https://www.youtube.com/watch?v=IpYmz3_BUM0&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=7
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

validation_value = 10000
x_value = train_data[:validation_value]  # Validation data.
x_train = train_data[validation_value:]

y_value = train_labels[:validation_value]
y_train = train_labels[validation_value:]

# https://www.youtube.com/watch?v=IpYmz3_BUM0&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=7
epochs = 40
fitModel = model.fit(x_train, y_train, epochs=epochs, batch_size=512, validation_data=(x_value, y_value), verbose=1)
results = model.evaluate(test_data, test_labels)
model.save("model.h5")  # .h5 is a extension that keras and tensorflow use.
"""

model = keras.models.load_model("model.h5")

test_review = input("Enter your review: ")
predict = model.predict(np.array([encode_review(test_review)]))
print(f"Sentence: {test_review}")
print(f"Prediction: {predict[0]}")
