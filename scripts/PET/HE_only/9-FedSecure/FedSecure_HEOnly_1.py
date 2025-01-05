# FedSecure Final
#  In the provided script, the FedSecure logic is reflected in the homomorphic_model_update function and the average_encrypted_models function.
# This function takes the global model weights, encrypts them using homomorphic encryption, adds the encrypted local model weights to the encrypted global model, and returns the updated encrypted model. This process reflects the FedSecure approach of aggregating model updates securely in the encrypted domain.
# This function calculates the average of the encrypted model weights provided by the federated clients. While the aggregation happens in the encrypted domain, the process ensures that the confidentiality of the individual model updates is maintained, which aligns with the FedSecure approach.

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

# Lists to store metrics
metrics_data = []

# Define a simple Keras model
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    return model

# Encryption Function
def encrypt_data(context, data):
    return ts.ckks_vector(context, data)

# Homomorphic Model Update Function for FedSecure
def homomorphic_model_update(context, global_model, encrypted_local_model):
    global_model_encrypted = encrypt_data(context, global_model)
    updated_model_encrypted = global_model_encrypted + encrypted_local_model
    return updated_model_encrypted

# Federated Average Function
def average_encrypted_models(context, encrypted_models):
    num_models = len(encrypted_models)

    # Aggregate encrypted model weights
    aggregated_model = sum(encrypted_models)

    return aggregated_model

# Load Dataset from Filesystem
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)  # Change this line based on your dataset format and loading method
        return dataset
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

try:
    # Create a TenSEAL context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

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
                    loss, accuracy, tp, tn, fp, fn = evaluation_metrics[0], evaluation_metrics[1], evaluation_metrics[2], evaluation_metrics[3], evaluation_metrics[4], evaluation_metrics[5]

                    # Update epoch-level metrics
                    round_metrics[f'Client {i+1} Epoch Time (seconds)'] = epoch_time
                    round_metrics[f'Client {i+1} Training Loss'] = loss
                    round_metrics[f'Client {i+1} Training Accuracy'] = accuracy
                    round_metrics[f'Client {i+1} True Positives'] = tp
                    round_metrics[f'Client {i+1} True Negatives'] = tn
                    round_metrics[f'Client {i+1} False Positives'] = fp
                    round_metrics[f'Client {i+1} False Negatives'] = fn

                    local_model_weights = local_model.get_weights()[0].flatten()
                    local_model_encrypted = encrypt_data(context, local_model_weights)

                    local_models.append(local_model_encrypted)

                # Aggregation
                start_time = time.time()
                average_model = average_encrypted_models(context, local_models)
                global_model_updated = homomorphic_model_update(
                    context, np.zeros(input_size), average_model
                )
                end_time = time.time()
                aggregation_time = end_time - start_time

                print("\nUpdated global model weights:", global_model_updated)
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
metrics_df.to_csv('FedSecure_HEOnly_metrics.csv', index=False)
