import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

# Read training and test data files
#train = pd.read_csv("UNSW_NB15_testing-set.csv").values
train  = pd.read_csv("UNSW_NB15_testing-set.csv").values

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
train[:,2]= label_encoder.fit_transform(train[:,2])       # proto
train[:,3]= label_encoder.fit_transform(train[:,3])       # service
train[:,4]= label_encoder.fit_transform(train[:,4])       # state
train[:,-2]= label_encoder.fit_transform(train[:,-2])     # attack_cat


# put labels into y_train variable
Y_train = train[:,-1:].astype(float)
# Drop 'label' column
X_train = train[:,1:-1].astype(float)


# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape:",X_train.shape)
print("x_test shape:",X_val.shape)
print("y_train shape:",Y_train.shape)
print("y_test shape:",Y_val.shape)


latent_dim = 43

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid',kernel_initializer='RandomNormal'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(43, activation='sigmoid'),
      layers.Reshape((-1,43))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train,epochs=5,shuffle=True,batch_size=32)



#########
# import os
#
# import flwr as fl
# import tensorflow as tf
#
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import *
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder
#
# data = pd.read_csv("temp/UNSW_NB15_training-set.csv")
# X=data.iloc[:,0:30]
# y=data.loc[:,['Class']]
#
# print(X)
# print("-----")
# print(y)
#
# # Make TensorFlow log less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#
#     model = Sequential()
#     model.add(Dense(10, input_dim=30, activation='sigmoid'))
#     model.add(Dense(20, activation='sigmoid'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')

    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class xClient(fl.client.NumPyClient):
    def get_parameters(self,config):  # type: ignore
        return autoencoder.get_weights()

    def fit(self, parameters, config):  # type: ignore
        autoencoder = Autoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        autoencoder.fit(X_train, X_train,epochs=5,shuffle=True,batch_size=32)
        autoencoder.set_weights(parameters)
        #autoencoder.fit(X_train, y_train, epochs=1, batch_size=32)

        return autoencoder.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        autoencoder.set_weights(parameters)
        loss, accuracy = autoencoder.evaluate(X_train, X_train)
        return loss, len(X_train), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=xClient())
