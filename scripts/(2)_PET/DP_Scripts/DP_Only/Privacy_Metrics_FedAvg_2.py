# This script contains HE + FL + BotNetIoT + Measures of Privacy Guarantees - It gives epsilon values (see PPT for details)
# Next I will also build checks for data confidentiality + Secure aggregation + robustness to attacks

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
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

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
FL_rounds = 2  # Number of Federated Learning rounds

# Initialize a global model
global_model_updated = create_model()

# Encryption and Decryption Functions
def encrypt_data(context, data):
    return ts.ckks_vector(context, data)

def decrypt_result(context, result):
    return result.decrypt()

# Homomorphic Model Update Function
def homomorphic_model_update(context, global_model, encrypted_local_model):
    global_model_encrypted = encrypt_data(context, global_model)
    updated_model_encrypted = global_model_encrypted + encrypted_local_model
    return decrypt_result(context, updated_model_encrypted)

# Federated Averaging Function (FedAvg)
def average_encrypted_models(context, encrypted_models):
    num_models = len(encrypted_models)
    aggregated_model = sum(encrypted_models, ts.ckks_vector(context, [0.0] * input_size))
    averaged_model = aggregated_model * (1.0 / num_models)
    return averaged_model

# Load Dataset from Filesystem
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
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
        clients = 2
        local_models = []

        for round_num in range(FL_rounds):
            print(f"\nFederated Learning Round {round_num + 1}:")

            for i in range(clients):
                local_data = train_data.sample(frac=0.1)
                local_features = local_data.values
                local_labels = local_data['label'].values

                local_train_size = int(0.8 * len(local_data))
                local_train_data, local_val_data = local_data[:local_train_size], local_data[local_train_size:]

                local_model = create_model()

                start_time = time.time()

                history = local_model.fit(
                    local_train_data.values,
                    local_train_data['label'].values,
                    validation_data=(local_val_data.values, local_val_data['label'].values),
                    epochs=2,
                    verbose=1,
                )

                evaluation_metrics = local_model.evaluate(
                    local_val_data.values,
                    local_val_data['label'].values,
                    verbose=0,
                )

                loss, accuracy, tp, tn, fp, fn = evaluation_metrics[0], evaluation_metrics[1], evaluation_metrics[2], evaluation_metrics[3], evaluation_metrics[4], evaluation_metrics[5]

                end_time = time.time()
                convergence_time = end_time - start_time

                print(
                    f"  Client {i+1} - "
                    f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}"
                )
                print(f"  Convergence Time: {convergence_time:.4f} seconds")

                local_model_weights = local_model.get_weights()
                local_models.append(local_model_weights)

            start_time = time.time()

            aggregated_model_weights = [np.zeros_like(w) for w in local_models[0]]
            aggregated_optimizer_states = [np.zeros_like(s) for s in local_models[0]]

            for local_model_weights in local_models:
                for i, w in enumerate(local_model_weights):
                    aggregated_model_weights[i] += w

            aggregated_weights = [w / len(local_models) for w in aggregated_model_weights]

            global_model_updated.set_weights(aggregated_weights)

            end_time = time.time()
            aggregation_time = end_time - start_time

            # Compute privacy metrics using TensorFlow Privacy
            # Sample parameters for computing privacy
            num_examples = len(train_data)  # Total number of training examples
            noise_multiplier = 0.5  # Noise multiplier for differential privacy
            batch_size = 256  # Batch size
            epochs = 2  # Number of training epochs

            # Compute privacy
            epsilon, _ = tf_privacy.compute_dp_sgd_privacy(
                n=num_examples,
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                epochs=epochs,
                delta=1e-5,
            )

            print("\nUpdated global model weights:", global_model_updated)
            print(f"  Aggregation Time for FL Round {round_num + 1}: {aggregation_time:.4f} seconds")
            print(f"  Epsilon: {epsilon}")

except Exception as e:
    print(f"An error occurred: {e}")
