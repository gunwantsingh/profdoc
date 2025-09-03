import os

import flwr as fl
import tensorflow as tf


import pandas as pd
from keras.models import Sequential
from keras.layers import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("temp/creditcard2_2.csv")
X=data.iloc[:,0:30]
y=data.loc[:,['Class']]

print(X)
print("-----")
print(y)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    model = Sequential()
    model.add(Dense(10, input_dim=30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')

    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client
    class xClient(fl.client.NumPyClient):
        def get_parameters(self,config):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(X_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=xClient())
