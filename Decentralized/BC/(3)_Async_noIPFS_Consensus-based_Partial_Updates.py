# 4 Consensus based aggregation + BC + nonIPFS + Partial updates

import os
import hashlib
import json
import pandas as pd
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from web3 import Web3
from solcx import compile_source, install_solc

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Constants and Settings
NODE_COUNT = 5
OUTPUT_DIR = "./block_details"
WEIGHTS_DIR = "./model_weights"
BLOCKCHAIN_FILE = "blockchain.json"
SOLC_VERSION = '0.8.0'
GANACHE_PORTS = [8545, 8546, 8547, 8548, 8549]  # Unique ports for Ganache instances
node_data_paths = ["../../dataset_partitions/partition_1.csv", "../../dataset_partitions/partition_2.csv", "../../dataset_partitions/partition_3.csv", "../../dataset_partitions/partition_4.csv", "../../dataset_partitions/partition_5.csv"]

# FL training parameters
input_size = 24
output_size = 1
learning_rate = 0.1
epochs = 40
FL_rounds = 15

# Ensure directories are set up
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create separate directories for each node's weights
for i in range(NODE_COUNT):
    os.makedirs(os.path.join(WEIGHTS_DIR, f"node_{i}"), exist_ok=True)

# Metrics CSV setup
metrics_path = os.path.join(OUTPUT_DIR, 'Metrics_(3)_Consensus_Async_FedAvg_noIPFS.csv')
with open(metrics_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Node", "Epoch", "FL Round", "Accuracy", "Loss", "Epoch Time", "FL Round Time", "Convergence Time", "Confusion Matrix", "Contributing Nodes", "Miner Node", "Model Weights Hash", "Update Count", "Training Time per Node"])

class Block:
    def __init__(self, index, previous_hash, data, weights_hash):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data
        self.weights_hash = weights_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.data}{self.weights_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.difficulty = 4

    def add_block(self, new_block):
        if not self.chain:
            self.chain.append(self.create_genesis_block())
        new_block.previous_hash = self.chain[-1].hash
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Block", "none")

    def save_to_file(self):
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'previous_hash': block.previous_hash,
                'data': block.data,
                'weights_hash': block.weights_hash,
                'nonce': block.nonce,
                'hash': block.hash
            })
        with open(BLOCKCHAIN_FILE, 'w') as f:
            json.dump(chain_data, f, indent=4)

def train_model(data_path, model_id, initial_weights=None):
    try:
        start_time = time.time()
        data = pd.read_csv(data_path)
        features = data.drop(columns=['label'])
        labels = data['label']
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(features.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        if initial_weights:
            model.set_weights(initial_weights)
        history = model.fit(features, labels, epochs=epochs, verbose=1)
        weights_path = os.path.join(WEIGHTS_DIR, f"node_{model_id}", f"model_weights_epoch_{epochs}.weights.h5")
        model.save_weights(weights_path)
        epoch_time = time.time() - start_time
        accuracies = history.history['accuracy']
        losses = history.history['loss']
        preds = model.predict(features)
        cm = confusion_matrix(labels, np.round(preds))
        model_weights_hash = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
        return weights_path, model, epoch_time, accuracies, losses, cm.flatten(), model_weights_hash, model_id
    except Exception as e:
        print(f"Error training model for node {model_id}: {e}")
        return None, None, None, None, None, None, None, model_id

def aggregate_models(weights_paths):
    try:
        print("Aggregating models...")
        accumulated_weights = None
        count = 0
        start_time = time.time()
        for path in weights_paths:
            if path is not None:  # Ensure the path is valid
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(input_size,)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(output_size, activation='sigmoid')
                ])
                model.load_weights(path)
                if accumulated_weights is None:
                    accumulated_weights = model.get_weights()
                else:
                    for i in range(len(accumulated_weights)):
                        accumulated_weights[i] += model.get_weights()[i]
                count += 1

        if count > 0:
            aggregated_weights = [weight / count for weight in accumulated_weights]
            aggregated_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_size,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(output_size, activation='sigmoid')
            ])
            aggregated_model.set_weights(aggregated_weights)
            
            aggregated_weights_path = os.path.join(WEIGHTS_DIR, "aggregated_model_weights.weights.h5")
            aggregated_model.save_weights(aggregated_weights_path)

            aggregation_time = time.time() - start_time
            return aggregated_weights_path, aggregated_weights, aggregation_time
        else:
            print("No valid weights paths to aggregate.")
            return None, None, None
    except Exception as e:
        print(f"Error aggregating models: {e}")
        return None, None, None

def vote_for_miner(node_metrics):
    max_power = max(node_metrics.values())
    powerful_nodes = [node for node, power in node_metrics.items() if power == max_power]
    miner_node = powerful_nodes[0] if powerful_nodes else 0
    print(f"Miner for this round: Node {miner_node}")
    return miner_node

def asynchronous_training(node_data_paths):
    futures = []
    with ThreadPoolExecutor(max_workers=NODE_COUNT) as executor:
        for node_id in range(NODE_COUNT):
            futures.append(executor.submit(train_model, node_data_paths[node_id], node_id))
    return futures

def consensus_based_aggregation(model_paths, metrics):
    # Sort models by their accuracy in descending order
    sorted_models = sorted(zip(model_paths, metrics), key=lambda x: x[1], reverse=True)
    # Take the top 2 out of the first 3 received updates
    top_models = [model[0] for model in sorted_models[:3]]
    best_models = [model for model in sorted_models[:2]]
    return aggregate_models(top_models), best_models

# Initialize blockchain
blockchain = Blockchain()

# Web3 setup
web3_instances = [Web3(Web3.HTTPProvider(f'http://127.0.0.1:{port}')) for port in GANACHE_PORTS]

# Federated Learning process
for FL_round in range(FL_rounds):
    round_start_time = time.time()

    # Asynchronous model training
    trained_models = asynchronous_training(node_data_paths)
    
    node_metrics = {}
    all_weights_paths = []
    all_models = []
    all_epoch_times = []
    all_accuracies = []
    all_losses = []
    all_conf_matrices = []
    contributing_nodes = []
    convergence_times = []

    for future in as_completed(trained_models):
        result = future.result()
        if result[0] is not None:
            weights_path, model, epoch_time, accuracies, losses, conf_matrix, model_weights_hash, node_id = result
            node_metrics[node_id] = max(accuracies)
            all_weights_paths.append(weights_path)
            all_models.append(model)
            all_epoch_times.append(epoch_time)
            all_accuracies.append(accuracies)
            all_losses.append(losses)
            all_conf_matrices.append(conf_matrix)
            contributing_nodes.append(node_id)
            
            # Convergence time calculation
            convergence_time = epoch_time * len(accuracies)
            convergence_times.append(convergence_time)

    # Aggregating models based on consensus mechanism
    (aggregated_weights_path, aggregated_weights, aggregation_time), best_models = consensus_based_aggregation(all_weights_paths, [max(acc) for acc in all_accuracies])
    if not aggregated_weights_path:
        continue

    round_time = time.time() - round_start_time

    # Print FL round time and convergence time
    print(f"FL Round {FL_round + 1} completed. FL Round Time: {round_time:.2f} seconds.")
    for i, convergence_time in enumerate(convergence_times):
        print(f"Node {contributing_nodes[i]} Convergence Time: {convergence_time:.2f} seconds.")

    # Update models with aggregated weights
    for model_id, model in enumerate(all_models):
        model.set_weights(aggregated_weights)
        model.save_weights(os.path.join(WEIGHTS_DIR, f"node_{model_id}", f"updated_model_weights_round_{FL_round + 1}.weights.h5"))  # Save the updated model

    with open(aggregated_weights_path, "rb") as f:
        aggregated_weights_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Select miner through voting
    miner_node = vote_for_miner(node_metrics)
    aggregated_data = f"Aggregated model updated and mined in round {FL_round + 1} by node {miner_node}."
    aggregated_block = Block(len(blockchain.chain), blockchain.chain[-1].hash if blockchain.chain else "0", aggregated_data, aggregated_weights_hash)
    blockchain.add_block(aggregated_block)
    with open(metrics_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for node_id in contributing_nodes:
            for epoch in range(epochs):
                writer.writerow([node_id, epoch, FL_round + 1, max(all_accuracies[node_id]), min(all_losses[node_id]), all_epoch_times[node_id], round_time, convergence_times[node_id], all_conf_matrices[node_id], ", ".join(map(str, contributing_nodes)), miner_node, aggregated_weights_hash, len(all_weights_paths), all_epoch_times[node_id]])

blockchain.save_to_file()

# Print blockchain data for verification
for block in blockchain.chain:
    print(f"Block Index: {block.index}")
    print(f"Previous Hash: {block.previous_hash}")
    print(f"Data: {block.data}")
    print(f"Nonce: {block.nonce}")
    print(f"Weights Hash: {block.weights_hash}")
    print(f"Hash: {block.hash}\n")
