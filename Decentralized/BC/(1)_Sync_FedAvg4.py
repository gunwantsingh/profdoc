# Experiment 3 Unified Sync

import os
import hashlib
import json
import pandas as pd
import tensorflow as tf


from web3 import Web3
import time
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import numpy as np
import csv

# Constants and Settings
NODE_COUNT = 5
OUTPUT_DIR = "./block_details"
WEIGHTS_DIR = "./model_weights"
BLOCKCHAIN_FILE = "blockchain.json"
FL_CONTRACT_ADDRESS = 'deployed_contract_addresses.json'
SOLC_VERSION = '0.8.0'
GANACHE_PORTS = [8545, 8547, 8549, 8551, 8553]  # Adjusted for 5 nodes
node_data_paths = ["../partition_1.csv", "../partition_2.csv", "../partition_3.csv", "../partition_4.csv", "../partition_5.csv"]

# FL training parameters
input_size = 24
output_size = 1
learning_rate = 0.1
epochs = 40  # Fixed number of epochs
FL_rounds = 15
convergence_threshold = 0.01
consecutive_epochs_for_convergence = 3

# Ensure directories are set up
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics CSV setup
metrics_path = os.path.join(OUTPUT_DIR, 'fl_metrics.csv')
with open(metrics_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Node", "Epoch", "FL Round", "Accuracy", "Loss", "Epoch Time", "FL Round Time", "Confusion Matrix", "F1 Score", "AUC-ROC", "Convergence Time"])

class Block:
    def __init__(self, index, previous_hash, data, weights_hash, weights_path):
        self.index = index
        self.previous_hash = self.hash if previous_hash is None else previous_hash
        self.data = data
        self.weights_hash = weights_hash
        self.weights_path = weights_path
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.data}{self.weights_hash}{self.weights_path}{self.nonce}"
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
            # Create a genesis block if not present
            self.chain.append(self.create_genesis_block())
        new_block.previous_hash = self.chain[-1].hash
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Block", "none", "none")

    def save_to_file(self):
        chain_data = [{
            'index': block.index,
            'previous_hash': block.previous_hash,
            'data': block.data,
            'weights_hash': block.weights_hash,
            'weights_path': block.weights_path,
            'nonce': block.nonce,
            'hash': block.hash
        } for block in self.chain]
        
        with open(BLOCKCHAIN_FILE, 'w') as f:
            json.dump(chain_data, f, indent=4)

def train_model(data_path, model_id, initial_weights=None):
    start_time = time.time()
    data = pd.read_csv(data_path)
    features = data.drop(columns=['label'])
    labels = data['label']
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    if initial_weights:
        model.set_weights(initial_weights)
    
    history = model.fit(features, labels, epochs=epochs, verbose=1)
    epoch_time = time.time() - start_time
    weights_path = os.path.join(WEIGHTS_DIR, f"node_{model_id}_model_weights.keras")
    model.save(weights_path)
    
    accuracies = history.history['accuracy']
    losses = history.history['loss']
    preds = model.predict(features)
    cm = confusion_matrix(labels, np.round(preds))
    f1 = f1_score(labels, np.round(preds))
    auc = roc_auc_score(labels, preds)
    
    return weights_path, model, epoch_time, accuracies, losses, cm.flatten(), f1, auc

def aggregate_models(weights_paths):
    print("Aggregating models...")
    accumulated_weights = None
    count = 0
    start_time = time.time()
    for path in weights_paths:
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
    
    aggregated_weights = [weight / count for weight in accumulated_weights]

    aggregated_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])
    aggregated_model.set_weights(aggregated_weights)
    
    aggregated_weights_path = os.path.join(WEIGHTS_DIR, "aggregated_model_weights.keras")
    aggregated_model.save(aggregated_weights_path)

    aggregation_time = time.time() - start_time
    return aggregated_weights_path, aggregated_weights, aggregation_time

# Initialize blockchain
blockchain = Blockchain()

for FL_round in range(FL_rounds):
    print(f"Federated Learning Round {FL_round + 1}")
    round_start_time = time.time()
    all_weights_paths = []
    all_models = []
    all_epoch_times = []
    all_accuracies = []
    all_losses = []
    all_conf_matrices = []
    all_f1_scores = []
    all_auc_scores = []
    for node_id in range(NODE_COUNT):
        weights_path, model, epoch_time, accuracies, losses, conf_matrix, f1, auc = train_model(node_data_paths[node_id], node_id)
        all_weights_paths.append(weights_path)
        all_models.append(model)
        all_epoch_times.append(epoch_time)
        all_accuracies.append(accuracies)
        all_losses.append(losses)
        all_conf_matrices.append(conf_matrix)
        all_f1_scores.append(f1)
        all_auc_scores.append(auc)
    
    aggregated_weights_path, aggregated_weights, aggregation_time = aggregate_models(all_weights_paths)
    round_time = time.time() - round_start_time

    # Log metrics for each epoch in each FL round, for each node
    with open(metrics_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for node_id in range(NODE_COUNT):
            for epoch in range(epochs):
                writer.writerow([node_id, epoch+1, FL_round+1, all_accuracies[node_id][epoch], all_losses[node_id][epoch], all_epoch_times[node_id], round_time, ', '.join(map(str, all_conf_matrices[node_id])), all_f1_scores[node_id], all_auc_scores[node_id], None])

    # Update models with aggregated weights
    for model_id, model in enumerate(all_models):
        model.set_weights(aggregated_weights)
        model.save(os.path.join(WEIGHTS_DIR, f"updated_model_weights_round_{FL_round + 1}_node_{model_id}.keras"))

    with open(aggregated_weights_path, "rb") as f:
        aggregated_weights_hash = hashlib.sha256(f.read()).hexdigest()
    
    aggregated_data = f"Aggregated model updated and mined in round {FL_round + 1}."
    aggregated_block = Block(len(blockchain.chain), blockchain.chain[-1].hash if blockchain.chain else "0", aggregated_data, aggregated_weights_hash, aggregated_weights_path)
    blockchain.add_block(aggregated_block)

blockchain.save_to_file()

# Print blockchain data for verification
for block in blockchain.chain:
    print(f"Block Index: {block.index}")
    print(f"Previous Hash: {block.previous_hash}")
    print(f"Data: {block.data}")
    print(f"Nonce: {block.nonce}")
    print(f"Weights Hash: {block.weights_hash}")
    print(f"Weights Path: {block.weights_path}")
    print(f"Hash: {block.hash}\n")
