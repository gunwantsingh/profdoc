import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Read training and test data files
#train = pd.read_csv("UNSW_NB15_testing-set.csv").values
train  = pd.read_csv("UNSW_NB15_training-set.csv").values
#train = pd.read_csv("/Users/gunwant/Desktop/_Gunwant/GCU/STAGE-2/Experiments/Datasets/UNSW-NB15/UNSW-NB15_2.csv")

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

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape:",X_train.shape)
print("x_test shape:",X_val.shape)
print("y_train shape:",Y_train.shape)
print("y_test shape:",Y_val.shape)

X_train = X_train.reshape(-1, 1, 43)
X_val  = X_val.reshape(-1, 1, 43)
Y_train = Y_train.reshape(-1, 1, 1)
Y_val = Y_val.reshape(-1, 1, 1)


print("x_train shape:",X_train.shape)
print("x_test shape:",X_val.shape)
print("y_train shape:",Y_train.shape)
print("y_test shape:",Y_val.shape)

# Model
model = keras.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
#model.add(layers.LSTM(43,input_shape=(43,1),kernel_initializer=initializer))
# -> model.add(layers.LSTM(43,input_dim=43,kernel_initializer=initializer,activation='sigmoid'))
model.add(layers.LSTM(43, input_dim=43, kernel_initializer=initializer, kernel_regularizer=regularizer, activation='sigmoid'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(86))
# model.add(layers.Dense(86))
model.add(layers.Dense(10))

#print(model.summary())

# metrics
# tensorboard_callback = TensorBoard(log_dir="./logs/")
opt = keras.optimizers.Adam()
#opt = 'sgd' (gives almost same accuracy)
# sgd = SGD(learning_rate=0.01, momentum=0.8)

# old -> model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=opt,metrics=["accuracy"])
#from_logits = True signifies the values of the loss obtained by the model are not normalized and is basically used when we don't have any softmax function in our model.

# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=86, epochs=1)
# Hyperparameters:
# AF = ['relu', 'sigmoid', 'tanh']
# BATCH_SIZE = [16, 32, 48, 64, 80, 96, 112]
# LAYERS = [1, 2, 3, 4, 5]
# NODES = [8, 16, 24, 32]
# WEIGHTS_INIT = ['Zeros', 'ones', 'RandomUniform', 'RandomNormal', 'GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal']
# OPTIMIZER = ['adagrad', 'adam', 'rmsprop', 'sgd']
# LEARNING_RATE_VALUE = [0.001, 1.005, 0.01, 0.1, 0.5]
# LOSS_FUNCTION = ['mse', 'categorical_crossentropy','sparse_categorical_crossentropy']
# REGULARIZATION = ['None', 'l1', 'l2', 'l1 and l2']
# DROPOUT = [0.0, 0.1, 0.2, 0.5] #(0%, 10%, 20%, 50%)
# model.add(Dense(30, input_dim=30, activation='relu', kernel_initializer='RandomNormal'))   # Input layer, (Input_dim = Input columns)


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
        model.fit(X_train, Y_train,validation_data=(X_val, Y_val),epochs=10,shuffle=True,batch_size=48)
        #model.fit(X_train, y_train, epochs=1, batch_size=32)

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        print("evaluate ji")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, Y_val)
        print("loss = ", loss)
        print ("accuracy = [after average?]", accuracy)

        # y_pred = model.predict(X_val)
        # fpr, tpr, thresholds = roc_curve(Y_val, y_pred)
        # auc_score = auc(fpr, tpr)
        # print("AUC_SCORE = ", auc_score)
        #return auc_score
        ##
        return loss, len(X_val), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=xClient())
