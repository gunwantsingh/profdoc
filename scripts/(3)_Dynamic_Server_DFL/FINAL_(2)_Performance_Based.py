import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time

# Function to load dataset
def load_dataset(partition):
    try:
        return pd.read_csv(f"partition_{partition}.csv")
    except FileNotFoundError:
        print(f"Error: partition_{partition}.csv not found.")
        return None

# Clean data by removing or replacing NaN and infinite values
def clean_data(data):
    return data.replace([np.inf, -np.inf], np.nan).dropna()

# Function to create Keras model for binary classification
def create_model(input_shape=(24,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.0001), metrics=['accuracy'])
    return model

# Function to monitor system resources
def monitor_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

# Function to calculate gradient norms
def calculate_gradient_norms(model, X_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(X_train, training=True)
        predictions = tf.clip_by_value(predictions, 1e-7, 1 - 1e-7)  # Clip predictions to avoid extreme values
        loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
    avg_gradient_norm = np.mean(gradient_norms)
    return avg_gradient_norm

# Function for local training on a node with convergence check
def local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, server_node):
    try:
        if X_train is None or y_train is None or X_test is None or y_test is None:
            return None

        print(f"Node {node_id + 1}: Local training started for Epoch {epoch} of FL Round {fl_round}...")

        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        # Reshape y_train and y_test to match the shape of model output
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        start_time = time.time()
        history = local_model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)  # Train for 1 epoch
        end_time = time.time()

        accuracy = history.history['accuracy'][0]
        loss = history.history['loss'][0]
        val_accuracy = history.history['val_accuracy'][0]
        val_loss = history.history['val_loss'][0]

        if np.isnan(loss) or np.isinf(loss):
            print(f"Node {node_id + 1}: Encountered NaN or Inf in loss. Skipping this step.")
            return None, None

        y_pred = local_model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        avg_gradient_norm = calculate_gradient_norms(local_model, X_train, y_train)

        print(f"Node {node_id + 1}: Local training completed with accuracy: {accuracy:.4f}, loss: {loss:.4f}, val_accuracy: {val_accuracy:.4f}, val_loss: {val_loss:.4f}, gradient norm: {avg_gradient_norm:.4f}, time: {end_time - start_time:.2f}s")

        # Capture node metrics including resource usage
        cpu_usage, memory_usage = monitor_resources()
        node_metrics.append([
            fl_round, epoch, node_id, accuracy, loss, val_accuracy, val_loss,
            fp, tp, fn, tn, avg_gradient_norm, end_time - start_time, server_node + 1, cpu_usage, memory_usage
        ])

        return local_model.get_weights(), loss
    except Exception as e:
        print(f"Error occurred during local training on Node {node_id + 1}: {e}")
        return None, None

# Function to aggregate models
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
    most_powerful_node = resource_usage[0][2]  # Corrected to access the third element of the first tuple
    print(f"Most powerful node: Node {most_powerful_node + 1}")
    return most_powerful_node

# Function to perform federated learning with convergence check and performance-based voting
def federated_learning(nodes_data, num_epochs, num_fl_rounds):
    try:
        all_node_metrics = []
        total_fl_round_times = []  # List to store FL round times for all rounds
        convergence_times = []  # List to store convergence times after each FL round

        # Create global model
        global_model = create_model()

        start_time = time.time()  # Start overall convergence time

        # Federated learning rounds
        for fl_round in range(1, num_fl_rounds + 1):
            fl_round_start_time = time.time()
            print(f"--- Federated Learning Round {fl_round} ---")

            # Determine server node based on resources before each FL round
            most_powerful_node = find_most_powerful_node(nodes_data)
            new_server_node = most_powerful_node
            print(f"Server role assigned to Node {new_server_node + 1} for FL round {fl_round}")

            node_metrics = []
            local_weights_list = []
            loss_history = []

            # Perform local training for each epoch sequentially with convergence check
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

                converged = True
                for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
                    weights, loss = local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, new_server_node)
                    if weights is not None:
                        local_weights_list.append(weights)
                        loss_history.append(loss)

                    # Convergence check: 5 consecutive loss values that are close
                    if len(loss_history) >= 5 and not np.allclose(loss_history[-5:], loss_history[-1], atol=1e-3):
                        converged = False

                if converged and len(loss_history) >= 5:
                    print(f"Convergence reached at Epoch {epoch} of FL Round {fl_round}. Moving to the next FL round.")
                    break

                print(f"Epoch {epoch} of Federated Learning Round {fl_round} completed.")

            # Aggregation step
            aggregated_weights = aggregate_models(local_weights_list)
            if aggregated_weights:
                global_model.set_weights(aggregated_weights)

            # Record FL round time and convergence time
            fl_round_end_time = time.time()
            fl_round_time = fl_round_end_time - fl_round_start_time
            total_fl_round_times.append(fl_round_time)
            convergence_times.append(fl_round_time)  # Since convergence stops the round, FL round time equals convergence time

            # Aggregate node metrics
            all_node_metrics.extend(node_metrics)

        # Total convergence time
        total_convergence_time = time.time() - start_time

        # Convert node metrics to DataFrame
        columns = [
            'FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss', 'Val_Accuracy', 'Val_Loss', 'FP', 'TP', 'FN', 'TN',
            'Gradient_Norm', 'Epoch_Time', 'Server_Node', 'CPU_Usage', 'Memory_Usage'
        ]
        node_metrics_df = pd.DataFrame(all_node_metrics, columns=columns)

        # Add FL_Round_Time and Convergence_Time columns to DataFrame
        fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
        convergence_times_repeated = np.repeat(convergence_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]

        node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
        node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

        # Calculate and add additional metrics
        node_metrics_df['Model_Size'] = global_model.count_params() * 4 / (1024 ** 2)  # Model size in MB (assuming 4 bytes per param)
        node_metrics_df['Model_Complexity'] = global_model.count_params()

        # Save to CSV
        node_metrics_df.to_csv('Final_Dynamic_P2P_FedAvg_Final.csv', index=False)

        print("\nTotal FL round times for all rounds (seconds):")
        for round_num, round_time in enumerate(total_fl_round_times, start=1):
            print(f"Round {round_num}: {round_time:.2f} seconds")

        print(f"\nTotal convergence time for the entire process: {total_convergence_time:.2f} seconds")
        print("Federated learning process completed.")

    except Exception as e:
        print(f"Error occurred during federated learning: {e}")

# Load datasets for each node
nodes_data = []
for partition in range(1, 6):
    data = load_dataset(partition)
    if data is not None and 'label' in data.columns:
        data = clean_data(data)
        X = data.drop(columns=['label']).values
        y = data['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        nodes_data.append((X_train, y_train, X_test, y_test))
    else:
        print(f"Skipping partition_{partition} due to missing data or missing 'label' column.")

# Run federated learning with the loaded datasets
federated_learning(nodes_data, num_epochs=40, num_fl_rounds=15)



# import numpy as np
# import pandas as pd
# import psutil
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Input
# from keras.optimizers import SGD
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import time
# import tensorflow as tf

# # Function to load dataset
# def load_dataset(partition):
#     try:
#         return pd.read_csv(f"partition_{partition}.csv")
#     except FileNotFoundError:
#         print(f"Error: partition_{partition}.csv not found.")
#         return None

# # Function to create Keras model for binary classification
# def create_model(input_shape=(24,)):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
#     return model

# # Function to monitor system resources
# def monitor_resources():
#     cpu_usage = psutil.cpu_percent(interval=1)
#     memory_info = psutil.virtual_memory()
#     memory_usage = memory_info.percent
#     return cpu_usage, memory_usage

# # Function to calculate gradient norms
# def calculate_gradient_norms(model, X_train, y_train):
#     with tf.GradientTape() as tape:
#         predictions = model(X_train, training=True)
#         loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
#     avg_gradient_norm = np.mean(gradient_norms)
#     return avg_gradient_norm

# # Function for local training on a node with convergence check
# # Function for local training on a node with convergence check
# def local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, server_node):
#     try:
#         if X_train is None or y_train is None or X_test is None or y_test is None:
#             return None

#         print(f"Node {node_id + 1}: Local training started for Epoch {epoch} of FL Round {fl_round}...")

#         local_model = create_model()
#         local_model.set_weights(global_model.get_weights())

#         # Reshape y_train to match the shape of model output
#         y_train = y_train.reshape(-1, 1)
#         y_test = y_test.reshape(-1, 1)

#         start_time = time.time()
#         history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
#         end_time = time.time()

#         accuracy = local_model.evaluate(X_test, y_test, verbose=0)[1]
#         y_pred = local_model.predict(X_test)
#         y_pred_binary = (y_pred > 0.5).astype(int)
#         cm = confusion_matrix(y_test, y_pred_binary)
#         tn, fp, fn, tp = cm.ravel()

#         avg_gradient_norm = calculate_gradient_norms(local_model, X_train, y_train)

#         print(f"Node {node_id + 1}: Local training completed with accuracy: {accuracy:.4f}, loss: {history.history['loss'][0]:.4f}, gradient norm: {avg_gradient_norm:.4f}, time: {end_time - start_time:.2f}s")

#         # Capture node metrics including resource usage
#         loss_values = history.history['loss'][0]  # Get loss values for the epoch
#         cpu_usage, memory_usage = monitor_resources()
#         node_metrics.append([fl_round, epoch, node_id, accuracy, loss_values, fp, tp, fn, tn, avg_gradient_norm, end_time - start_time, server_node + 1, cpu_usage, memory_usage])

#         return local_model.get_weights(), loss_values
#     except Exception as e:
#         print(f"Error occurred during local training on Node {node_id + 1}: {e}")
#         return None, None


# # Function to aggregate models
# def aggregate_models(weights_list):
#     if not weights_list:
#         return None

#     num_layers = len(weights_list[0])
#     aggregated_weights = []
#     for layer_idx in range(num_layers):
#         layer_weights = np.mean([weights[layer_idx] for weights in weights_list], axis=0)
#         aggregated_weights.append(layer_weights)

#     return aggregated_weights

# # Function to find the most powerful node based on resources
# def find_most_powerful_node(nodes_data):
#     resource_usage = []
#     for node_id in range(len(nodes_data)):
#         cpu_usage, mem_usage = monitor_resources()
#         resource_usage.append((cpu_usage, mem_usage, node_id))
#         print(f"Node {node_id + 1} - CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage}%")
#     # Sort nodes by CPU usage and memory usage (lower is better)
#     resource_usage.sort(key=lambda x: (x[0], x[1]))
#     most_powerful_node = resource_usage[0][2]  # Corrected to access the third element of the first tuple
#     print(f"Most powerful node: Node {most_powerful_node + 1}")
#     return most_powerful_node

# # Function to perform federated learning with convergence check and performance-based voting
# def federated_learning(nodes_data, num_epochs, num_fl_rounds):
#     try:
#         all_node_metrics = []
#         total_fl_round_times = []  # List to store FL round times for all rounds
#         convergence_times = []  # List to store convergence times after each FL round

#         # Create global model
#         global_model = create_model()

#         start_time = time.time()  # Start overall convergence time

#         # Federated learning rounds
#         for fl_round in range(1, num_fl_rounds + 1):
#             fl_round_start_time = time.time()
#             print(f"--- Federated Learning Round {fl_round} ---")

#             # Determine server node based on resources before each FL round
#             most_powerful_node = find_most_powerful_node(nodes_data)  # Corrected
#             new_server_node = most_powerful_node  # Corrected
#             print(f"Server role assigned to Node {new_server_node + 1} for FL round {fl_round}")

#             node_metrics = []
#             local_weights_list = []
#             loss_history = []

#             # Perform local training for each epoch sequentially with convergence check
#             for epoch in range(1, num_epochs + 1):
#                 print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

#                 converged = True
#                 for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
#                     weights, loss = local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, new_server_node)
#                     if weights is not None:
#                         local_weights_list.append(weights)
#                         loss_history.append(loss)

#                     if len(loss_history) >= 5 and not all(np.isclose(loss_history[-5:], loss_history[-1])):
#                         converged = False

#                 if converged and len(loss_history) >= 5:
#                     print(f"Convergence reached at Epoch {epoch} of FL Round {fl_round}. Moving to the next FL round.")
#                     break

#                 print(f"Epoch {epoch} of Federated Learning Round {fl_round} completed.")

#             # Aggregation step
#             aggregated_weights = aggregate_models(local_weights_list)
#             if aggregated_weights:
#                 global_model.set_weights(aggregated_weights)

#             # Record FL round time and convergence time
#             fl_round_end_time = time.time()
#             fl_round_time = fl_round_end_time - fl_round_start_time
#             total_fl_round_times.append(fl_round_time)
#             convergence_times.append(fl_round_time)  # Since convergence stops the round, FL round time equals convergence time

#             # Aggregate node metrics
#             all_node_metrics.extend(node_metrics)

#         # Total convergence time
#         total_convergence_time = time.time() - start_time

#         # Convert node metrics to DataFrame
#         columns = [
#             'FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss', 'FP', 'TP', 'FN', 'TN', 'Gradient_Norm', 'Epoch_Time', 'Server_Node',
#             'CPU_Usage', 'Memory_Usage'
#         ]
#         node_metrics_df = pd.DataFrame(all_node_metrics, columns=columns)

#         # Add FL_Round_Time and Convergence_Time columns to DataFrame
#         fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
#         convergence_times_repeated = np.repeat(convergence_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]

#         node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
#         node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

#         # Calculate and add additional metrics
#         node_metrics_df['Model_Size'] = global_model.count_params() * 4 / (1024 ** 2)  # Model size in MB (assuming 4 bytes per param)
#         node_metrics_df['Model_Complexity'] = global_model.count_params()

#         # Save to CSV
#         node_metrics_df.to_csv('Final_Dynamic_P2P_FedAvg_Final.csv', index=False)

#         print("\nTotal FL round times for all rounds (seconds):")
#         for round_num, round_time in enumerate(total_fl_round_times, start=1):
#             print(f"Round {round_num}: {round_time:.2f} seconds")

#         print(f"\nTotal convergence time for the entire process: {total_convergence_time:.2f} seconds")
#         print("Federated learning process completed.")

#     except Exception as e:
#         print(f"Error occurred during federated learning: {e}")

# # Load datasets for each node
# nodes_data = []
# for partition in range(1, 6):
#     data = load_dataset(partition)
#     if data is not None and 'label' in data.columns:
#         X = data.drop(columns=['label']).values
#         y = data['label'].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         nodes_data.append((X_train, y_train, X_test, y_test))
#     else:
#         print(f"Skipping partition_{partition} due to missing data or missing 'label' column.")

# # Run federated learning with the loaded datasets
# federated_learning(nodes_data, num_epochs=40, num_fl_rounds=15)
