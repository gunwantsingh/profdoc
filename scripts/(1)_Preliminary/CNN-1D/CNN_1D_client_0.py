import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K
from keras import layers
import tensorflow as tf

import flwr as fl

from tensorflow import keras
from keras import layers, losses
from keras.models import Model
from keras import initializers
from keras import regularizers
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing # Import label encoder
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

#####

# Read training and test data files
train = pd.read_csv("UNSW_NB15_testing-set.csv").values
test  = pd.read_csv("UNSW_NB15_training-set.csv").values

train[:, 1:-1]

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
X_train

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape:",X_train.shape)
print("x_test shape:",X_val.shape)
print("y_train shape:",Y_train.shape)
print("y_test shape:",Y_val.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(43,)),               #Flatten layer will come before Dense Layer, always
    tf.keras.layers.Dense(16,  activation='relu'),
    tf.keras.layers.Reshape((16, 1)), # ADD THIS LINE OF CODE
    tf.keras.layers.Conv1D(16, kernel_size=(2), activation='relu', padding='same'),     # Convolutional Layer, responsible for Feature extraction
    tf.keras.layers.MaxPooling1D(pool_size=(4), strides=3, padding='valid'),            # Pooling Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')        # Fully connected layer
])
model.summary()

# optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=["accuracy"])
epochs = 10  # for better result increase the epochs
batch_size = 250

# Fit the model
# model.fit(X_train, Y_train,epochs=epochs,batch_size= batch_size)
# score = model.evaluate(X_test, y_test, batch_size=128)

tensorboard_callback = TensorBoard(log_dir="./logs/")
opt = keras.optimizers.Adam()

# Define Flower client
class xClient(fl.client.NumPyClient):
    def get_parameters(self,config):  # type: ignore
        print("get params")
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        #model = model(latent_dim)
        print("hello")
        model.set_weights(parameters)
        # old -> history = model.fit(X_train, Y_train,epochs=20,shuffle=True,batch_size=48, callbacks=[tensorboard_callback])
        model.fit(X_train, Y_train,epochs=epochs,batch_size= batch_size, callbacks=[tensorboard_callback])

        #model.fit(X_train, y_train, epochs=1, batch_size=32)

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        print("evaluate ji")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, Y_val)
        print("loss = ", loss)
        print ("accuracy = ", accuracy)
        # y_pred = model.predict(X_val)
        # fpr, tpr, thresholds = roc_curve(Y_val, y_pred)
        # auc_score = auc(fpr, tpr)
        # print("AUC_SCORE = ", auc_score)
        #return auc_score
        ##
        return loss, len(X_val), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=xClient())

















