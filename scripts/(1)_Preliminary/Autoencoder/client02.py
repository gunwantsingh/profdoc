import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
RANDOM_SEED = 2021
TEST_PCT = 0.3
LABELS = ["Normal","Fraud"]
##
import flwr as fl

from tensorflow import keras
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

# Read training and test data files
dataset = pd.read_csv("creditcard2.csv")
train = pd.read_csv("UNSW_NB15_testing-set.csv").values
test  = pd.read_csv("UNSW_NB15_training-set.csv").values

# Import label encoder
from sklearn import preprocessing

### ///
normal_dataset = dataset[dataset.Class == 0]
fraud_dataset = dataset[dataset.Class == 1]

sc=StandardScaler()
dataset['Time'] = sc.fit_transform(dataset['Time'].values.reshape(-1, 1))
dataset['Amount'] = sc.fit_transform(dataset['Amount'].values.reshape(-1, 1))

raw_data = dataset.values
# The last element contains if the transaction is normal which is represented by a 0 and if fraud then 1
labels = raw_data[:, -1]
# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=2021
)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
#creating normal and fraud datasets
normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]
fraud_train_data = train_data[train_labels]
fraud_test_data = test_data[test_labels]

nb_epoch = 50
batch_size = 64
input_dim = normal_train_data.shape[1] #num of columns, 30
encoding_dim = 14
hidden_dim_1 = int(encoding_dim / 2) #
hidden_dim_2=4
learning_rate = 1e-7


#input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))
#Encoder
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
encoder=tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)
# Decoder
decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
decoder=tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)
#Autoencoder
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()
#print(autoencoder.get_weights())

### ///
#
# # label_encoder object knows how to understand word labels.
# label_encoder = preprocessing.LabelEncoder()
#
# # Encode labels in column 'species'.
# train[:,2]= label_encoder.fit_transform(train[:,2])       # proto
# train[:,3]= label_encoder.fit_transform(train[:,3])       # service
# train[:,4]= label_encoder.fit_transform(train[:,4])       # state
# train[:,-2]= label_encoder.fit_transform(train[:,-2])     # attack_cat
#
#
# # put labels into y_train variable
# Y_train = train[:,-1:].astype(float)
# # Drop 'label' column
# X_train = train[:,1:-1].astype(float)
#
#
# # Split the train and the validation set for the fitting
# from sklearn.model_selection import train_test_split
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
# print("x_train shape:",X_train.shape)
# print("x_test shape:",X_val.shape)
# print("y_train shape:",Y_train.shape)
# print("y_test shape:",Y_val.shape)
#
#
# latent_dim = 43
#
# class Autoencoder(Model):
#   def __init__(self, latent_dim):
#     super(Autoencoder, self).__init__()
#     self.latent_dim = latent_dim
#     self.encoder = tf.keras.Sequential([
#       layers.Flatten(),
#       layers.Dense(latent_dim, activation='relu'),
#     ])
#     self.decoder = tf.keras.Sequential([
#       layers.Dense(43, activation='relu'),
#       layers.Reshape((-1,43))
#     ])
#
#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded
#
# autoencoder = Autoencoder(latent_dim)
#
# autoencoder.compile(optimizer='sgd', metrics='accuracy', loss='mse')
#
# autoencoder.fit(X_train, X_train,epochs=10,shuffle=True)

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
#     model.add(Dense(10, input_dim=30, activation='relu'))
#     model.add(Dense(20, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')

    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class xClient(fl.client.NumPyClient):
    def get_parameters(self,config):  # type: ignore
        return autoencoder.get_weights()

    def fit(self, parameters, config):  # type: ignore
        autoencoder.set_weights(parameters)
        #autoencoder.fit(X_train, y_train, epochs=1, batch_size=32)
        autoencoder.fit(normal_train_data, normal_train_data,epochs=5,shuffle=True,batch_size=32)
        return autoencoder.get_weights(), len(normal_train_data), {}

    def evaluate(self, parameters, config):  # type: ignore
        autoencoder.set_weights(parameters)
        loss, accuracy = autoencoder.evaluate(normal_train_data, normal_train_data)
        return loss, len(normal_train_data), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=xClient())
