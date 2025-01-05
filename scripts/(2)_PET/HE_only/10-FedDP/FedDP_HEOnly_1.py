# This example only uses FedDP as it adds noise (no HE)

import numpy as np
import pandas as pd
import tensorflow as tf
import tenseal as ts
import time

# Global model parameters
input_size = 25
output_size = 1
learning_rate = 0.1
FL_rounds = 10  # Number of Federated Learning rounds

# Differential Privacy parameters
epsilon = 1.0
delta = 1e-5

# Lists to store metrics
metrics_data = []

# Define a simple Keras model
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load Dataset from Filesystem
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)  # Change this line based on your dataset format and loading method
        return dataset
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

# Differential Privacy Mechanism
def add_noise(weights, epsilon, delta):
    noise = np.random.laplace(scale = 2.0 / epsilon, size=weights.shape)
    return weights + noise

try:
    # Load the dataset from the filesystem
    file_path = "BotNetIoT.csv"  # Update with the actual path to your dataset file
    loaded_dataset = load_dataset(file_path)

    if loaded_dataset is not None:
        # Split the dataset into training (80%) and validation (20%) sets
        train_size = int(0.8 * len(loaded_dataset))
        train_data, val_data = loaded_dataset[:train_size], loaded_dataset[train_size:]

        # Simulate three federated clients
        clients = 5

        for round_num in range(FL_rounds):
            print(f"\nFederated Learning Round {round_num + 1}:")

            round_start_time = time.time()  # Start time of FL round

            for epoch_num in range(30):
                round_metrics = {'FL round': round_num + 1, 'Epoch': epoch_num + 1}

                local_models = []  # Initialize local models list

                for i in range(clients):
                    local_data = train_data.sample(frac=0.1)
                    local_features = local_data.values
                    local_labels = local_data['label'].values

                    local_train_size = int(0.8 * len(local_data))
                    local_train_data, local_val_data = local_data[:local_train_size], local_data[local_train_size:]

                    local_model = create_model()

                    start_time = time.time()  # Record the start time

                    history = local_model.fit(
                        local_train_data.values,
                        local_train_data['label'].values,
                        validation_data=(local_val_data.values, local_val_data['label'].values),
                        # epochs=1,
                        verbose=1,
                    )

                    end_time = time.time()  # Record the end time
                    epoch_time = end_time - start_time  # Calculate epoch time

                    # Evaluate the model on validation data to get metrics
                    evaluation_metrics = local_model.evaluate(
                        local_val_data.values,
                        local_val_data['label'].values,
                        verbose=0,
                    )

                    # Extract the metrics from the evaluation results
                    loss, accuracy = evaluation_metrics[0], evaluation_metrics[1]

                    # Update epoch-level metrics
                    round_metrics[f'Client {i+1} Epoch Time (seconds)'] = epoch_time
                    round_metrics[f'Client {i+1} Training Loss'] = loss
                    round_metrics[f'Client {i+1} Training Accuracy'] = accuracy

                    local_model_weights = local_model.get_weights()[0].flatten()
                    local_model_weights_noisy = add_noise(local_model_weights, epsilon, delta)

                    local_models.append(local_model_weights_noisy)

                # Aggregation
                start_time = time.time()
                global_model = np.mean(local_models, axis=0)
                end_time = time.time()
                aggregation_time = end_time - start_time

                print("\nGlobal model weights after aggregation:", global_model)
                print(f"  Aggregation Time for FL Round {round_num + 1}, Epoch {epoch_num + 1}: {aggregation_time:.4f} seconds")

                # Update epoch-level metrics
                round_metrics['Server Aggregation Time (seconds)'] = aggregation_time

                metrics_data.append(round_metrics)

            round_end_time = time.time()  # End time of FL round
            round_time = round_end_time - round_start_time  # Calculate FL round time

            # Update round-level metrics
            round_metrics['FL Round Time (seconds)'] = round_time

except Exception as e:
    print(f"An error occurred: {e}")

# Save metrics to a CSV file
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('FedDP_HeOnly_metrics.csv', index=False)
