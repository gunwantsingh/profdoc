import flwr as fl
#import tensorflow as fl
import keras
from keras.models import *
from keras.layers import *
from keras import layers, losses
from typing import Dict, Optional, Tuple
from pathlib import Path
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

#NUM_CLIENTS = 2

model = Sequential()
#model.add(Dense(43, input_dim=43, activation='sigmoid')) #This one for Autoencoder
model.add(layers.LSTM(43,input_shape=(43,1)))
#model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
model.add(layers.Dense(10))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')


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

        #x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

        #//#

        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X_train,Y_train)
        return loss, {"accuracy": accuracy}

    return evaluate

#params = model.get_parameters()
# Working: FedAvg(99.87), FedAdagrad (00.13), Fedyogi(99.87), FedAvgM(48.77), FedOpt(.0013), FedAdam (00.13) (TODO: FedProx, QFedAvg, FedOptim - same as FedOpt?)
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
    # evaluate_metrics_aggregation_fn=weighted_average,
    # on_fit_config_fn=fit_config,
    #initial_parameters=fl.common.ndarrays_to_parameters(params)
)
while True:
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)
