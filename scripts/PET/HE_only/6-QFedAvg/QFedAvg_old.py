# QFedAvg HE WORKS FINE!!
# Added a quantization function
# To implement Quantized Federated Averaging (QFedAvg), you need to modify the aggregation step to quantize the model weights before averaging.
# This modification quantizes the model weights before averaging and encrypts the quantized weights before updating the global model. Keep in mind that the dequantization is performed after updating the global model.

import numpy as np
import pandas as pd
import tensorflow as tf
import tenseal as ts
import time
from sklearn.metrics import confusion_matrix

# Global model parameters
input_size = 51
output_size = 1
learning_rate = 0.1
FL_rounds = 2  # Number of Federated Learning rounds

# Define a simple Keras model
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    return model

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

# Quantization Function
def quantize_weights(weights, num_bits=8):
    max_value = np.max(np.abs(weights))
    scale = (2 ** (num_bits - 1) - 1) / max_value
    quantized_weights = np.round(weights * scale).astype(np.int32)
    return quantized_weights, scale

# Quantized Federated Averaging Function
def quantized_average_encrypted_models(context, encrypted_models):
    num_models = len(encrypted_models)

    # Extract and quantize model weights
    model_weights = [model.decrypt() for model in encrypted_models]
    quantized_weights, scale = quantize_weights(np.mean(np.array(model_weights), axis=0))

    # Encrypt the quantized weights
    quantized_model_encrypted = encrypt_data(context, quantized_weights.tolist())

    return quantized_model_encrypted, scale


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
    file_path = "../s1.csv"  # Update with the actual path to your dataset file
    loaded_dataset = load_dataset(file_path)

    if loaded_dataset is not None:
        # Split the dataset into training (80%) and validation (20%) sets
        train_size = int(0.8 * len(loaded_dataset))
        train_data, val_data = loaded_dataset[:train_size], loaded_dataset[train_size:]

        # Simulate three federated clients
        clients = 3
        local_models = []

        for round_num in range(FL_rounds):
            print(f"\nFederated Learning Round {round_num + 1}:")

            for i in range(clients):
                local_data = train_data.sample(frac=0.1)
                local_features = local_data.values
                local_labels = local_data['Security_Attack'].values

                local_train_size = int(0.8 * len(local_data))
                local_train_data, local_val_data = local_data[:local_train_size], local_data[local_train_size:]

                local_model = create_model()

                start_time = time.time()  # Record the start time

                history = local_model.fit(
                    local_train_data.values,
                    local_train_data['Security_Attack'].values,
                    validation_data=(local_val_data.values, local_val_data['Security_Attack'].values),
                    epochs=3,
                    verbose=1,
                )

                # Evaluate the model on validation data to get metrics
                evaluation_metrics = local_model.evaluate(
                    local_val_data.values,
                    local_val_data['Security_Attack'].values,
                    verbose=0,
                )

                # Extract the metrics from the evaluation results
                loss, accuracy, tp, tn, fp, fn = evaluation_metrics[0], evaluation_metrics[1], evaluation_metrics[2], evaluation_metrics[3], evaluation_metrics[4], evaluation_metrics[5]

                end_time = time.time()  # Record the end time
                convergence_time = end_time - start_time

                print(
                    f"  Client {i+1} - "
                    f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}"
                )
                print(f"  Convergence Time: {convergence_time:.4f} seconds")

                local_model_weights = local_model.get_weights()[0].flatten()
                local_model_encrypted = encrypt_data(context, local_model_weights)

                local_models.append(local_model_encrypted)

            # Aggregation with Quantization
            start_time = time.time()
            quantized_model, scale = quantized_average_encrypted_models(context, local_models)
            global_model_updated = homomorphic_model_update(
                context, np.zeros(input_size), quantized_model
            )
            end_time = time.time()
            aggregation_time = end_time - start_time

            print("\nUpdated global model weights (after dequantization):", global_model_updated / scale)
            print(f"  Aggregation Time for FL Round {round_num + 1}: {aggregation_time:.4f} seconds")





except Exception as e:
    print(f"An error occurred: {e}")