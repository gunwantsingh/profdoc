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


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

# Create strategy and run server
strategy = fl.server.strategy.AggregateCustomMetricStrategy(
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    agg = aggregate_evaluate(model)
)


fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)
