# Découverte réseaux de neurones
# Mois d'août 2019
# Avec le tutoriel de Thibault Neveu

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler #Pour la normalisation qui permet d'éviter des données trop "éparpillées"
# En effet par ex pixels de 0 à 255 --> donne un a priori au réseau puisque certains pèsent 255 fois plus que d'autres au départ
from sklearn.model_selection import train_test_split

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_,_) = fashion_mnist.load_data()
images = images[ : 10000]
targets = targets[ : 10000]


#Flatten
images = images.reshape(-1,784)
images = images.astype(float)

#Normalize
scaler = StandardScaler()
images = scaler.fit_transform(images)

images_train, images_test, targets_train, targets_test = train_test_split(images, targets, test_size = 0.2, random_state = 1)

#print(images.shape)
#print(targets.shape)
#print(images[0])
#print(targets[0])


########## Plot one of the data ##########
targets_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
#plt.imshow(images[7], cmap = "binary")
#plt.title(targets_names[targets[7]])
#print(targets[7])
#plt.show()

########## Flatten the image ##########

model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape = [28,28]))

#print("Shape of the image", images[0:2].shape)
#model_output = model.predict(images[0:1])
#print("Shape of the image after the Flatten", model_output.shape)

########## Add the layers ##########

model.add(tf.keras.layers.Dense(256, activation = "relu"))
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

#model_output = model.predict(images[0:5])
#print(model_output, targets[0:5])

#model.summary()

########## Define the loss function ##########
model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = "sgd",
        metrics = ["accuracy"]
        )

########### Train the model ##########
history = model.fit(images_train, targets_train, epochs = 10, validation_split = 0.2)

#loss_curve = history.history["loss"]
#acc_curve = history.history["acc"]
#loss_val_curve = history.history["val_loss"]
#acc_val_curve = history.history["val_acc"]
#
#plt.plot(loss_curve, label = "Train")
#plt.plot(loss_val_curve, label = "Val")
#plt.legend(loc = 'upper left')
##plt.title("Loss")
#plt.show()
#
#plt.plot(acc_curve, label = "Train")
#plt.plot(acc_val_curve, label = "Val")
#plt.legend(loc = 'upper left')
##plt.title("Accuracy")
#plt.show()

#model_output = model.predict(images[0:5])
#print(model_output, targets[0:5])

model.evaluate(images_test, targets_test)

model.save("simple_nm.h5")
