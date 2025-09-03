import flwr as fl
#import tensorflow as fl
import keras
from keras.models import *
from keras.layers import *
from keras import layers, losses
from  keras.callbacks import TensorBoard

from typing import Dict, Optional, Tuple
from pathlib import Path
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score

# tensorboard_callback = TensorBoard(log_dir="./logs/")
#NUM_CLIENTS = 2

# Model
model = keras.Sequential()
#initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
#regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
#model.add(layers.LSTM(43,input_shape=(43,1),kernel_initializer=initializer))
# -> model.add(layers.LSTM(43,input_dim=43,kernel_initializer=initializer,activation='sigmoid'))
model.add(layers.LSTM(43, input_dim=43, activation='sigmoid'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(86))
# model.add(layers.Dense(86))
model.add(layers.Dense(10))



def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        #//#

        train  = pd.read_csv("UNSW_NB15_testing-set.csv").values

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

        #//#
        X_train = X_train.reshape(-1, 1, 43)
        X_val  = X_val.reshape(-1, 1, 43)
        Y_train = Y_train.reshape(-1, 1, 1)
        Y_val = Y_val.reshape(-1, 1, 1)




        tensorboard_callback = TensorBoard(log_dir="./logs/")
        opt = keras.optimizers.Adam()
        #opt = 'sgd' (gives almost same accuracy)
        # sgd = SGD(learning_rate=0.01, momentum=0.8)

        # old -> model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=opt,metrics=["accuracy"])
        #model.fit(X_train, Y_train,validation_data=(X_val, Y_val),epochs=10,shuffle=True,batch_size=48, callbacks=[tensorboard_callback])



        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X_val,Y_val)

        #one, two, three, four = model.evaluate(x_val,y_val)
        return loss, {"accuracy": accuracy}
        #return one, two, three, four
    return evaluate

#params = model.get_parameters()
# Working: FedAvg(99.87), FedAdagrad (00.13), FedYogi(99.87), FedAvgM(48.77), FedOpt(.0013), FedAdam (00.13) (TODO: FedProx, QFedAvg, FedOptim - same as FedOpt?)
# FedAvg: (McMahan et al., 2017)
# FedProx: Li et al. (2020)
# QFedAvg: Li et al. (2019)
# FedOptim: Reddi et al. (2021)
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.025,  # Train on 25 clients (each round)
    # fraction_evaluate=0.05,  # Evaluate on 50 clients (each round)
    # min_fit_clients=2,
    # min_evaluate_clients=2,
    # min_available_clients=NUM_CLIENTS,
    # initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    evaluate_fn=get_evaluate_fn(model),
    #evaluate_metrics_aggregation_fn=weighted_average,
    # on_fit_config_fn=fit_config,
    #initial_parameters=fl.common.ndarrays_to_parameters(params)
)
while True:
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)
