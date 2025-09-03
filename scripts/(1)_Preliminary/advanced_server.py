from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import tensorflow as tf
from sklearn import preprocessing

import keras
from keras.models import *
from keras.layers import *
from keras import layers, losses


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Sequential()
    #model.add(Dense(43, input_dim=43, activation='sigmoid')) #This one for Autoencoder
    model.add(layers.LSTM(43,input_shape=(43,1)))
    #model.add(Dense(20, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

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


    # Split the train and the validation set for the fitting
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()