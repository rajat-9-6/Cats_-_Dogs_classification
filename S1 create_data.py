import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

DataDir = r"C:\Users\Rajat\PycharmProjects\MachineLearning\Project - Classification of Dogs & Cats\PetImages/"

Categories = ["Dog", "Cat"]

for i in Categories:
    path = os.path.join(DataDir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

img_size = 100

new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap="gray")
plt.show()

training_data = []


def create_training_data():
    for i in Categories:
        path = os.path.join(DataDir, i)
        class_num = Categories.index(i)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


create_training_data()
print(len(training_data))

import random

random.shuffle(training_data)

# for sample in training_data[:10]:
#     # print(sample)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

# print(X[0].reshape(-1, img_size, img_size, 1))

X = np.array(X).reshape(-1, img_size, img_size, 1)

import pickle

pickle_out = open("C:/Users/Rajat/PycharmProjects/MachineLearning/Project - Classification of Dogs & Cats/X.pickle",
                  "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("C:/Users/Rajat/PycharmProjects/MachineLearning/Project - Classification of Dogs & Cats/Y.pickle",
                  "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
