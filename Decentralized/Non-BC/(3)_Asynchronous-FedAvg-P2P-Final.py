# Asynchronous + P2P with queues + Decentralized + metrics (w/ aggr time) + no threading


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import queue

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

# Function for local training on a node
def local_training(node_id, X_train, y_train, X_test, y_test, local_model, update_queues, epoch, fl_round):
    try:
        if X_train is None or y_train is None or X_test is None or y_test is None:
            return None

        print(f"Node {node_id}: Local training started...")

        start_time = time.time()
        history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
        end_time = time.time()

        accuracy = local_model.evaluate(X_test, y_test, verbose=0)[1]
        y_pred = local_model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        print(f"Node {node_id}: Local training completed with accuracy: {accuracy}")

        # Capture node metrics
        loss_values = history.history['loss'][0]
        node_metrics = [fl_round, epoch, node_id, accuracy, loss_values, fp, tp, fn, tn, end_time - start_time]

        # Send local model update to other nodes via queues
        for neighbor_id, queue in update_queues.items():
            if neighbor_id != node_id:  # Skip sending update to self
                queue.put((local_model.get_weights(), node_id))

        return local_model, node_metrics
    except Exception as e:
        print(f"Error occurred during local training: {e}")
        return None

# Function to update models based on received weights and capture aggregation time
def update_model(local_model, update_queue):
    aggregation_start_time = time.time()
    local_weights = []
    while not update_queue.empty():
        weights, sender_id = update_queue.get()
        local_weights.append(weights)

    if not local_weights:
        return 0

    new_weights = []
    num_layers = len(local_model.get_weights())

    for layer_idx in range(num_layers):
        layer_weights = np.mean([weights[layer_idx] for weights in local_weights], axis=0)
        new_weights.append(layer_weights)

    local_model.set_weights(new_weights)
    aggregation_end_time = time.time()

    return aggregation_end_time - aggregation_start_time

# Function to perform federated learning
def federated_learning(nodes_data, num_epochs, num_fl_rounds):
    try:
        all_node_metrics = []

        # Initialize models for each node
        local_models = [create_model(input_shape=nodes_data[0][0].shape[1:]) for _ in range(len(nodes_data))]
        update_queues = {node_id: queue.Queue() for node_id in range(len(nodes_data))}

        total_fl_round_times = []
        total_aggregation_times = []
        start_time = time.time()

        for fl_round in range(1, num_fl_rounds + 1):
            fl_round_start_time = time.time()
            print(f"--- Federated Learning Round {fl_round} ---")

            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

                for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
                    result = local_training(node_id, X_train, y_train, X_test, y_test, local_models[node_id], update_queues, epoch, fl_round)
                    if result:
                        local_model, node_metrics = result
                        local_models[node_id] = local_model
                        all_node_metrics.append(node_metrics)

                # Each node updates its local model asynchronously and captures aggregation time
                for node_id, update_queue in update_queues.items():
                    aggregation_time = update_model(local_models[node_id], update_queue)
                    total_aggregation_times.append((fl_round, epoch, node_id, aggregation_time))
                    print(f"Node {node_id} aggregation time for epoch {epoch} in FL round {fl_round}: {aggregation_time}")

            fl_round_end_time = time.time()
            fl_round_time = fl_round_end_time - fl_round_start_time
            total_fl_round_times.append(fl_round_time)

        # Capture convergence time
        convergence_time = time.time() - start_time

        # Convert node metrics to DataFrame
        node_metrics_df = pd.DataFrame(all_node_metrics, columns=['FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss', 'FP', 'TP', 'FN', 'TN', 'Epoch_Time'])

        # Debugging: Check the DataFrame before adding additional columns
        print(node_metrics_df.head())

        # Create a DataFrame for aggregation times
        aggregation_times_df = pd.DataFrame(total_aggregation_times, columns=['FL_Round', 'Epoch', 'Node_ID', 'Aggregation_Time'])

        # Merge aggregation times with node metrics DataFrame
        node_metrics_df = node_metrics_df.merge(aggregation_times_df, on=['FL_Round', 'Epoch', 'Node_ID'], how='left')

        # Add FL_Round_Time and Convergence_Time columns to DataFrame
        fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
        convergence_times_repeated = np.repeat(convergence_time, len(node_metrics_df))

        node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
        node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

        node_metrics_df.to_csv('Asynchronous_FedAvg_P2P_Final.csv', index=False)

        print("\nTotal FL round times for all rounds (seconds):")
        for round_num, round_time in enumerate(total_fl_round_times, start=1):
            print(f"Round {round_num}: {round_time}")

        print("\nTotal aggregation times for all rounds (seconds):")
        for round_num, aggregation_time in enumerate(total_aggregation_times, start=1):
            print(f"Round {round_num}: {aggregation_time[3]}")

        print(f"\nTotal convergence time (seconds): {convergence_time}")

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








# OLD:

# import numpy as np
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# from keras.models import Sequential
# from keras.layers import Dense, Input
# from keras.optimizers import SGD
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import time
# import queue
# import threading

# # Function to load dataset
# def load_dataset(partition):
#     try:
#         return pd.read_csv(f"../../dataset_partitions/partition_{partition}.csv")
#     except FileNotFoundError:
#         print(f"Error: partition_{partition}.csv not found.")
#         return None

# # Function to create Keras model for binary classification
# def create_model(input_shape=(24,)):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
#     return model

# # Function for local training on a node
# def local_training(node_id, X_train, y_train, X_test, y_test, local_model, update_queues, epoch, fl_round):
#     try:
#         if X_train is None or y_train is None or X_test is None or y_test is None:
#             return None

#         print(f"Node {node_id}: Local training started...")

#         start_time = time.time()
#         history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
#         end_time = time.time()

#         accuracy = local_model.evaluate(X_test, y_test, verbose=0)[1]
#         y_pred = local_model.predict(X_test)
#         y_pred_binary = (y_pred > 0.5).astype(int)
#         cm = confusion_matrix(y_test, y_pred_binary)
#         tn, fp, fn, tp = cm.ravel()

#         print(f"Node {node_id}: Local training completed with accuracy: {accuracy}")

#         # Capture node metrics
#         loss_values = history.history['loss'][0]
#         thread_id = threading.get_ident()
#         node_metrics = [fl_round, epoch, node_id, accuracy, loss_values, fp, tp, fn, tn, end_time - start_time, thread_id]

#         # Send local model update to other nodes via queues
#         for neighbor_id, queue in update_queues.items():
#             if neighbor_id != node_id:  # Skip sending update to self
#                 queue.put((local_model.get_weights(), node_id))

#         return local_model, node_metrics
#     except Exception as e:
#         print(f"Error occurred during local training: {e}")
#         return None

# # Function to update models based on received weights and capture aggregation time
# def update_model(local_model, update_queue):
#     aggregation_start_time = time.time()
#     local_weights = []
#     while not update_queue.empty():
#         weights, sender_id = update_queue.get()
#         local_weights.append(weights)

#     if not local_weights:
#         return 0

#     new_weights = []
#     num_layers = len(local_model.get_weights())

#     for layer_idx in range(num_layers):
#         layer_weights = np.mean([weights[layer_idx] for weights in local_weights], axis=0)
#         new_weights.append(layer_weights)

#     local_model.set_weights(new_weights)
#     aggregation_end_time = time.time()

#     return aggregation_end_time - aggregation_start_time

# # Function to perform federated learning
# def federated_learning(nodes_data, num_epochs, num_fl_rounds):
#     try:
#         all_node_metrics = []

#         # Initialize models for each node
#         local_models = [create_model(input_shape=nodes_data[0][0].shape[1:]) for _ in range(len(nodes_data))]
#         update_queues = {node_id: queue.Queue() for node_id in range(len(nodes_data))}

#         total_fl_round_times = []
#         total_aggregation_times = []
#         start_time = time.time()

#         for fl_round in range(1, num_fl_rounds + 1):
#             fl_round_start_time = time.time()
#             print(f"--- Federated Learning Round {fl_round} ---")

#             for epoch in range(1, num_epochs + 1):
#                 print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

#                 with ThreadPoolExecutor(max_workers=len(nodes_data)) as executor:
#                     futures = [
#                         executor.submit(local_training, node_id, X_train, y_train, X_test, y_test, local_models[node_id], update_queues, epoch, fl_round)
#                         for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data)
#                     ]

#                     for future in futures:
#                         result = future.result()
#                         if result:
#                             local_model, node_metrics = result
#                             local_models[node_metrics[2]] = local_model
#                             all_node_metrics.append(node_metrics)

#                 # Each node updates its local model asynchronously and captures aggregation time
#                 for node_id, update_queue in update_queues.items():
#                     aggregation_time = update_model(local_models[node_id], update_queue)
#                     total_aggregation_times.append((fl_round, epoch, node_id, aggregation_time))
#                     print(f"Node {node_id} aggregation time for epoch {epoch} in FL round {fl_round}: {aggregation_time}")

#             fl_round_end_time = time.time()
#             fl_round_time = fl_round_end_time - fl_round_start_time
#             total_fl_round_times.append(fl_round_time)

#         # Capture convergence time
#         convergence_time = time.time() - start_time

#         # Convert node metrics to DataFrame
#         node_metrics_df = pd.DataFrame(all_node_metrics, columns=['FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss', 'FP', 'TP', 'FN', 'TN', 'Epoch_Time', 'Thread_ID'])

#         # Debugging: Check the DataFrame before adding additional columns
#         print(node_metrics_df.head())

#         # Create a DataFrame for aggregation times
#         aggregation_times_df = pd.DataFrame(total_aggregation_times, columns=['FL_Round', 'Epoch', 'Node_ID', 'Aggregation_Time'])

#         # Merge aggregation times with node metrics DataFrame
#         node_metrics_df = node_metrics_df.merge(aggregation_times_df, on=['FL_Round', 'Epoch', 'Node_ID'], how='left')

#         # Add FL_Round_Time and Convergence_Time columns to DataFrame
#         fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
#         convergence_times_repeated = np.repeat(convergence_time, len(node_metrics_df))

#         node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
#         node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

#         node_metrics_df.to_csv('Asynchronous_FedAvg_P2P_Final.csv', index=False)

#         print("\nTotal FL round times for all rounds (seconds):")
#         for round_num, round_time in enumerate(total_fl_round_times, start=1):
#             print(f"Round {round_num}: {round_time}")

#         print("\nTotal aggregation times for all rounds (seconds):")
#         for round_num, aggregation_time in enumerate(total_aggregation_times, start=1):
#             print(f"Round {round_num}: {aggregation_time[3]}")

#         print(f"\nTotal convergence time (seconds): {convergence_time}")

#     except Exception as e:
#         print(f"Error occurred during federated learning: {e}")

# # Load datasets for each node
# nodes_data = []
# for partition in range(1, 6):
#     data = load_dataset(partition)
#     if data is not None and 'label' in data.columns:
#         print(f"Dataset partition_{partition} shape:", data.shape)
#         X = data.drop('label', axis=1)
#         y = data['label']  # Assuming 'label' is already encoded as binary (0 or 1)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         nodes_data.append((X_train, y_train, X_test, y_test))
#     else:
#         print(f"Error loading partition_{partition}.csv. Skipping...")

# # Perform federated learning
# num_epochs = 40
# num_fl_rounds = 15
# federated_learning(nodes_data, num_epochs, num_fl_rounds)
