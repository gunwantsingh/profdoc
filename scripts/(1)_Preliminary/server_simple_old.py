import flwr as fl
#import tensorflow as fl
import keras
from keras.models import *
from keras.layers import *
from keras import layers, losses
from typing import Dict, Optional, Tuple
from pathlib import Path

#NUM_CLIENTS = 2

model = Sequential()
#model.add(Dense(43, input_dim=43, activation='sigmoid')) #This one for Autoencoder
model.add(layers.LSTM(43,input_shape=(43,1)))
#model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
model.add(layers.Dense(10))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}
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
    evaluate_fn=evaluate,
    # evaluate_metrics_aggregation_fn=weighted_average,
    # on_fit_config_fn=fit_config,
    #initial_parameters=fl.common.ndarrays_to_parameters(params)
)
while True:
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)
