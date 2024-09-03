import numpy as np
import pandas as pd
import psutil
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time

# Function to load dataset
def load_dataset(partition):
    try:
        return pd.read_csv(f"../../dataset_partitions/partition_{partition}.csv")
    except FileNotFoundError:
        print(f"Error: partition_{partition}.csv not found.")
        return None

# Function to create Keras model for binary classification
def create_model(input_shape=(24,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
    return model

# Function to monitor system resources
def monitor_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

# Function for local training on a node
def local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, server_node):
    try:
        if X_train is None or y_train is None or X_test is None or y_test is None:
            return None

        print(f"Node {node_id}: Local training started...")

        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        start_time = time.time()
        history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
        end_time = time.time()

        accuracy = local_model.evaluate(X_test, y_test, verbose=0)[1]
        y_pred = local_model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        print(f"Node {node_id}: Local training completed with accuracy: {accuracy}")

        # Capture node metrics including resource usage
        loss_values = history.history['loss'][0]  # Get loss values for the epoch
        node_metrics.append([fl_round, epoch, node_id, accuracy, loss_values, fp, tp, fn, tn, end_time - start_time, server_node + 1])

        return local_model.get_weights()
    except Exception as e:
        print(f"Error occurred during local training: {e}")
        return None

# Function to update models based on received weights
def aggregate_models(weights_list):
    if not weights_list:
        return None

    num_layers = len(weights_list[0])
    aggregated_weights = []
    for layer_idx in range(num_layers):
        layer_weights = np.mean([weights[layer_idx] for weights in weights_list], axis=0)
        aggregated_weights.append(layer_weights)

    return aggregated_weights

# Function to find the most powerful node based on resources
def find_most_powerful_node(nodes_data):
    resource_usage = []
    for node_id in range(len(nodes_data)):
        cpu_usage, mem_usage = monitor_resources()
        resource_usage.append((cpu_usage, mem_usage, node_id))
        print(f"Node {node_id + 1} - CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage}%")
    # Sort nodes by CPU usage and memory usage (lower is better)
    resource_usage.sort(key=lambda x: (x[0], x[1]))
    most_powerful_node = resource_usage[0][2]
    print(f"Most powerful node: Node {most_powerful_node + 1}")
    return resource_usage

# Function to perform federated learning
def federated_learning(nodes_data, num_epochs, num_fl_rounds):
    try:
        all_node_metrics = []

        # Create global model
        global_model = create_model()

        total_fl_round_times = []  # List to store FL round times for all rounds
        convergence_times = []  # List to store convergence times after each FL round

        start_time = time.time()  # Start overall convergence time

        # Federated learning rounds
        for fl_round in range(1, num_fl_rounds + 1):
            fl_round_start_time = time.time()
            print(f"--- Federated Learning Round {fl_round} ---")

            # Determine server node based on resources before each FL round
            post_round_resource_usage = find_most_powerful_node(nodes_data)
            new_server_node = post_round_resource_usage[0][2]
            print(f"Server role assigned to Node {new_server_node + 1} for FL round {fl_round}")

            node_metrics = []
            local_weights_list = []

            # Perform local training for each epoch sequentially
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

                for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
                    weights = local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, new_server_node)
                    if weights:
                        local_weights_list.append(weights)

                print(f"Epoch {epoch} of Federated Learning Round {fl_round} completed.")

            # Aggregation step
            aggregated_weights = aggregate_models(local_weights_list)
            if aggregated_weights:
                global_model.set_weights(aggregated_weights)

            # Record FL round time
            fl_round_end_time = time.time()
            fl_round_time = fl_round_end_time - fl_round_start_time
            total_fl_round_times.append(fl_round_time)

            # Record convergence time after each FL round
            current_convergence_time = fl_round_end_time - start_time
            convergence_times.append(current_convergence_time)

            # Aggregate node metrics
            all_node_metrics.extend(node_metrics)

        # Total convergence time
        total_convergence_time = time.time() - start_time

        # Convert node metrics to DataFrame
        columns = ['FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss', 'FP', 'TP', 'FN', 'TN', 'Epoch_Time', 'Server_Node']
        node_metrics_df = pd.DataFrame(all_node_metrics, columns=columns)

        # Add FL_Round_Time and Convergence_Time columns to DataFrame
        fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
        convergence_times_repeated = np.repeat(convergence_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]

        node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
        node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

        # Save to CSV
        node_metrics_df.to_csv('Dynamic_P2P_FedAvg_Final.csv', index=False)

        print("\nTotal FL round times for all rounds (seconds):")
        for round_num, round_time in enumerate(total_fl_round_times, start=1):
            print(f"Round {round_num}: {round_time}")

        print(f"\nTotal convergence time (seconds): {total_convergence_time}")

    except Exception as e:
        print(f"Error occurred during federated learning: {e}")

# Load datasets for each node
nodes_data = []
for partition in range(1, 6):
    data = load_dataset(partition)
    if data is not None and 'label' in data.columns:
        print(f"Dataset partition_{partition} shape:", data.shape)
        X = data.drop('label', axis=1)
        y = data['label']  # Assuming 'label' is already encoded as binary (0 or 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        nodes_data.append((X_train, y_train, X_test, y_test))
    else:
        print(f"Error loading partition_{partition}.csv. Skipping...")

# Perform federated learning
num_epochs = 40
num_fl_rounds = 15
federated_learning(nodes_data, num_epochs, num_fl_rounds)
