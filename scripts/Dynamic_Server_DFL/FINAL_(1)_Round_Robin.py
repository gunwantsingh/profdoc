import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf

# Function to load dataset
def load_dataset(partition):
    try:
        return pd.read_csv(f"partition_{partition}.csv")
    except FileNotFoundError:
        print(f"Error: partition_{partition}.csv not found.")
        return None

# Function to create Keras model for binary classification
def create_model(input_shape=(24,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
    return model

# Function to compute gradients manually
def compute_gradients(model, X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss = model.compute_loss(X, y, y_pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    total_gradient_norm = np.sum([np.linalg.norm(grad.numpy()) for grad in gradients])
    return total_gradient_norm

# Function for local training on a node with additional metrics
def local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, server_node):
    try:
        if X_train is None or y_train is None or X_test is None or y_test is None:
            return None

        print(f"Node {node_id + 1}: Local training started for Epoch {epoch} of FL Round {fl_round}...")

        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        start_time = time.time()
        gradient_before = compute_gradients(local_model, X_train, y_train)
        history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
        end_time = time.time()
        gradient_after = compute_gradients(local_model, X_train, y_train)

        gradient_diminishment = gradient_before - gradient_after
        accuracy = local_model.evaluate(X_test, y_test, verbose=0)[1]
        y_pred = local_model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        model_complexity = local_model.count_params()

        print(f"Node {node_id + 1}: Local training completed for Epoch {epoch} with accuracy: {accuracy:.4f}, loss: {history.history['loss'][0]:.4f}, time: {end_time - start_time:.2f}s")

        # Staleness of updates (time since last FL round)
        staleness = end_time - start_time if fl_round > 1 else 0

        # Capture node metrics including resource usage
        loss_values = history.history['loss'][0]  # Get loss values for the epoch
        node_metrics.append([
            fl_round, epoch, node_id, accuracy, loss_values,
            fp, tp, fn, tn, end_time - start_time, server_node + 1,
            gradient_diminishment, model_complexity, staleness
        ])

        return local_model.get_weights(), loss_values
    except Exception as e:
        print(f"Error occurred during local training on Node {node_id + 1}: {e}")
        return None, None

# Function to aggregate models
def aggregate_models(weights_list):
    new_weights = []
    num_layers = len(weights_list[0])
    for layer_idx in range(num_layers):
        layer_weights = np.mean([weights[layer_idx] for weights in weights_list], axis=0)
        new_weights.append(layer_weights)
    return new_weights

# Function to check if the model has converged based on loss
def check_convergence(loss_history, tolerance=1e-4):
    if len(loss_history) < 5:
        return False
    recent_losses = loss_history[-5:]
    return max(recent_losses) - min(recent_losses) < tolerance

# Function to perform federated learning
def federated_learning(nodes_data, num_epochs, num_fl_rounds, X_val, y_val):
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

            # Round-robin assignment of aggregator node
            aggregator_node = (fl_round - 1) % len(nodes_data)
            print(f"Aggregator role assigned to Node {aggregator_node + 1} for FL round {fl_round}")

            node_metrics = []
            local_weights_list = []
            loss_history = []

            # Perform local training for each epoch sequentially
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch} of Federated Learning Round {fl_round} started...")

                for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
                    weights, loss = local_training(node_id, X_train, y_train, X_test, y_test, global_model, node_metrics, epoch, fl_round, aggregator_node)
                    if weights is not None:
                        local_weights_list.append(weights)
                    if loss is not None:
                        loss_history.append(loss)

                print(f"Epoch {epoch} of Federated Learning Round {fl_round} completed.")

                # Check for convergence
                if check_convergence(loss_history):
                    print(f"Convergence reached in FL Round {fl_round} after {epoch} epochs.")
                    break  # Exit the loop if convergence is reached

            # Aggregation step
            aggregated_weights = aggregate_models(local_weights_list)
            global_model.set_weights(aggregated_weights)

            # Calculate validation accuracy and loss
            val_loss, val_accuracy = global_model.evaluate(X_val, y_val, verbose=0)

            print(f"Validation Accuracy after FL Round {fl_round}: {val_accuracy:.4f}")
            print(f"Validation Loss after FL Round {fl_round}: {val_loss:.4f}")

            # Record convergence time after aggregation
            fl_round_end_time = time.time()
            current_convergence_time = fl_round_end_time - fl_round_start_time
            convergence_times.append(current_convergence_time)

            # Record FL round time
            fl_round_time = fl_round_end_time - fl_round_start_time
            total_fl_round_times.append(fl_round_time)

            # Aggregate node metrics and add validation metrics
            for metric in node_metrics:
                metric.extend([val_accuracy, val_loss])

            all_node_metrics.extend(node_metrics)

        # Total convergence time
        total_convergence_time = time.time() - start_time

        # Convert node metrics to DataFrame
        columns = [
            'FL_Round', 'Epoch', 'Node_ID', 'Accuracy', 'Loss',
            'FP', 'TP', 'FN', 'TN', 'Epoch_Time', 'Aggregator_Node',
            'Gradient_Diminishment', 'Model_Complexity', 'Staleness',
            'Validation_Accuracy', 'Validation_Loss'
        ]
        node_metrics_df = pd.DataFrame(all_node_metrics, columns=columns)

        # Add FL_Round_Time and Convergence_Time columns to DataFrame
        fl_round_times_repeated = np.repeat(total_fl_round_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]
        convergence_times_repeated = np.repeat(convergence_times, num_epochs * len(nodes_data))[:len(node_metrics_df)]

        node_metrics_df['FL_Round_Time'] = pd.Series(fl_round_times_repeated)
        node_metrics_df['Convergence_Time'] = pd.Series(convergence_times_repeated)

        # Save to CSV
        node_metrics_df.to_csv('Final_RoundRobin_FedAvg_Final.csv', index=False)

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

# Create a validation dataset from one of the partitions (e.g., partition_1)
# You can use part of partition_1's training set as the validation set
X_val, X_train_split, y_val, y_train_split = train_test_split(nodes_data[0][0], nodes_data[0][1], test_size=0.8, random_state=42)

# Update partition_1 training data with the new split
nodes_data[0] = (X_train_split, y_train_split, nodes_data[0][2], nodes_data[0][3])

# Perform federated learning
num_epochs = 40
num_fl_rounds = 15
federated_learning(nodes_data, num_epochs, num_fl_rounds, X_val, y_val)
