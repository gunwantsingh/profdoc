import numpy as np
import pandas as pd
import tensorflow as tf
import tenseal as ts
import time
from sklearn.metrics import confusion_matrix
import tensorflow_privacy as tf_privacy

# Define a simple Keras model
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_main)  # Add a hidden layer
    hidden_layer2 = tf.keras.layers.Dense(32, activation='relu')(hidden_layer1)  # Add another hidden layer
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(hidden_layer2)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)

    # Switch to Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    return model

# Global model parameters
input_size=25
output_size = 1
learning_rate = 0.1
FL_rounds = 10  # Number of Federated Learning rounds
epochs = 30  # Number of training epochs

# Initialize a global model
global_model_updated = create_model()

# Encryption and Decryption Functions
def encrypt_data(context, data):
    return ts.ckks_vector(context, data)

def decrypt_result(context, result):
    return result.decrypt()

# Load Dataset from Filesystem
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

# Method to compute privacy guarantees
def compute_privacy(train_data, num_examples, batch_size, noise_multiplier, epochs):
    try:
        # Compute privacy metrics using TensorFlow Privacy
        epsilon, _ = tf_privacy.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=1e-5,
        )

        return epsilon
    except Exception as e:
        print(f"An error occurred while computing privacy: {e}")
        return None

def fault_tolerant_aggregate(local_models):
    # Fault Tolerance Mechanism
    if len(local_models) < clients:
        print("Not enough models received. Using previous global model.")
    else:
        aggregated_model_weights = [np.zeros_like(w) for w in local_models[0]]
        aggregated_optimizer_states = [np.zeros_like(s) for s in local_models[0]]

        for local_model_weights in local_models:
            for i, w in enumerate(local_model_weights):
                aggregated_model_weights[i] += w

        aggregated_weights = [w / len(local_models) for w in aggregated_model_weights]

        global_model_updated.set_weights(aggregated_weights)

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
        local_models = []

        # Create an empty list to store the metrics
        metrics = []

        for round_num in range(FL_rounds):
            round_metrics = []

            print(f"\nFederated Learning Round {round_num + 1}:")
            round_start_time = time.time()

            for i in range(clients):
                local_data = train_data.sample(frac=0.1)
                local_features = local_data.values
                local_labels = local_data['label'].values

                local_train_size = int(0.8 * len(local_data))
                local_train_data, local_val_data = local_data[:local_train_size], local_data[local_train_size:]

                local_model = create_model()

                local_model_start_time = time.time()

                for epoch in range(epochs):
                    epoch_start_time = time.time()

                    history = local_model.fit(
                        local_train_data.values,
                        local_train_data['label'].values,
                        validation_data=(local_val_data.values, local_val_data['label'].values),
                        epochs=1,
                        verbose=1,
                    )

                    epoch_end_time = time.time()
                    epoch_time = epoch_end_time - epoch_start_time

                    print(f"    Client {i+1} Epoch {epoch+1} Training Time: {epoch_time:.4f} seconds")

                local_model_end_time = time.time()
                local_model_convergence_time = local_model_end_time - local_model_start_time

                evaluation_metrics = local_model.evaluate(
                    local_val_data.values,
                    local_val_data['label'].values,
                    verbose=0,
                )

                loss, accuracy, tp, tn, fp, fn = evaluation_metrics[0], evaluation_metrics[1], evaluation_metrics[2], evaluation_metrics[3], evaluation_metrics[4], evaluation_metrics[5]

                print(
                    f"  Client {i+1} - "
                    f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, "
                    f"Convergence Time: {local_model_convergence_time:.4f} seconds"
                )

                local_model_weights = local_model.get_weights()
                local_models.append(local_model_weights)

                # Store metrics for this client
                round_metrics.append({
                    'Node': i + 1,
                    'FL Round': round_num + 1,
                    'Epoch': epochs,  # Only store the last epoch
                    'Training Time': local_model_convergence_time,
                    'Loss': loss,
                    'Accuracy': accuracy,
                    'TP': tp,
                    'TN': tn,
                    'FP': fp,
                    'FN': fn,
                })

            aggregation_start_time = time.time()

            fault_tolerant_aggregate(local_models)

            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            # Compute privacy
            epsilon = compute_privacy(train_data, len(train_data), batch_size=256, noise_multiplier=0.5, epochs=3)

            round_end_time = time.time()
            round_time = round_end_time - round_start_time

            print(f"  Aggregation Time for FL Round {round_num + 1}: {aggregation_time:.4f} seconds")
            print(f"  Round Time: {round_time:.4f} seconds")
            print(f"  Epsilon: {epsilon}")

            # Store metrics for this round
            round_metrics.append({
                'Aggregation Time': aggregation_time,
                'Round Time': round_time,
                'Epsilon': epsilon
            })

            # Append round metrics to the overall metrics list
            metrics.extend(round_metrics)

        # Convert metrics list to DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Save metrics to CSV file
        metrics_df.to_csv('metrics.csv', index=False)

except Exception as e:
    print(f"An error occurred: {e}")
