# Final FedAvM
#
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
momentum = 0.9  # Momentum parameter for FedAvgM

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

# Homomorphic Model Update Function for FedAvgM
def homomorphic_model_update(global_model, local_model, global_momentum):
    updated_model = (global_model * global_momentum) + ((1 - global_momentum) * local_model)
    return updated_model

# Federated Average Function
def average_encrypted_models(encrypted_models):
    num_models = len(encrypted_models)
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

        global_model = np.zeros(input_size)  # Initialize global model with zeros

        for round_num in range(FL_rounds):
            print(f"\nFederated Learning Round {round_num + 1}:")

            round_start_time = time.time()  # Start time of FL round

            for epoch_num in range(30):
                round_metrics = {'FL round': round_num + 1, 'Epoch': epoch_num + 1}  # Initialize round_metrics here

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
                        epochs=1,
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

                    # Update global model using FedAvgM
                    global_model = homomorphic_model_update(
                        global_model, local_model_weights, momentum
                    )

                # End of local training loop

                metrics_data.append(round_metrics)  # Append round_metrics to metrics_data after each epoch

            # End of epoch loop

            round_end_time = time.time()  # End time of FL round
            round_time = round_end_time - round_start_time  # Calculate FL round time

            # Update round-level metrics
            round_metrics['FL Round Time (seconds)'] = round_time

            # metrics_data.append(round_metrics)  # Removed this line as it's not needed here

        # End of FL round loop

except Exception as e:
    print(f"An error occurred: {e}")

# Save metrics to a CSV file
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('FedAvgM_HEOnly_metrics.csv', index=False)
