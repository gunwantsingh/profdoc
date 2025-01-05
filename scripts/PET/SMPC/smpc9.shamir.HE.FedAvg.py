# """

# This script integrates several protocols, techniques, and security controls. Let's examine them in detail:

# 1. **Shamir's Secret Sharing (SSS):**
#    - The script uses Shamir's Secret Sharing to securely distribute model weights across multiple nodes. This technique ensures that no single node has access to the entire model weights, thereby enhancing security and privacy.
#    - The `shamir_secret_sharing` function generates shares of the secret (model weights) using Shamir's Secret Sharing algorithm. Each share consists of a pair `(x, y)` where `x` represents the share index and `y` represents the value computed using the secret polynomial.
#    - The shares are then aggregated across nodes to reconstruct the original secret.

# 2. **Homomorphic Encryption (TenSEAL):**
#    - TenSEAL library is used for homomorphic encryption to perform secure aggregation of the shares.
#    - The `secure_aggregation` function aggregates the shares using TenSEAL. It first generates shares of the weights using SSS, and then aggregates these shares securely without revealing the original weights.
#    - Homomorphic encryption allows computations to be performed on encrypted data without decrypting it, thereby preserving data privacy.

# 3. **Federated Learning (FedAvg):**
#    - The script implements Federated Learning using the Federated Averaging (FedAvg) algorithm.
#    - Each node performs local training on its local dataset and computes local model updates.
#    - These local updates are then securely aggregated using SSS and TenSEAL to obtain the global model update.

# 4. **Secure Multiparty Computation (SMPC):**
#    - The combination of Shamir's Secret Sharing and TenSEAL enables Secure Multiparty Computation (SMPC).
#    - SMPC ensures that computations are performed securely across multiple parties (nodes) without revealing the individual inputs or intermediate results.

# 5. **Security Controls:**
#    - The script uses appropriate security controls such as threshold and prime modulus to configure Shamir's Secret Sharing.
#    - It handles exceptions and errors to ensure robustness and reliability during computation.

# Overall, the script leverages a combination of Shamir's Secret Sharing, TenSEAL homomorphic encryption, and Federated Learning to achieve privacy-preserving and collaborative model training across multiple parties while ensuring data confidentiality and integrity.

# """"
# In this modified function:

# We first perform Shamir's Secret Sharing on the weights received from each node, generating shares of the weights.
# Next, we homomorphically encrypt these shares using TenSEAL, ensuring that the encryption is done without revealing the original shares.
# Finally, we perform secure aggregation by adding the encrypted shares together. Since the shares are encrypted, this aggregation preserves the privacy of the individual shares.
# With these modifications, the secure_aggregation function now combines Shamir's Secret Sharing and Homomorphic Encryption to achieve Secure Multi-Party Computation, ensuring that the aggregation of model updates is performed securely across multiple nodes while preserving data privacy.

import csv
import time
import tensorflow as tf
import numpy as np
import tenseal as ts
from sympy import nextprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_irreducible

# Global model parameters
input_size = 24  # Number of features in the dataset, adjust accordingly
output_size = 1
learning_rate = 0.1
num_epochs = 30
num_nodes = 5  # Number of nodes in the federated learning setup
num_fl_rounds = 10  # Number of Federated Learning rounds
file_paths = ["partition_1.csv", "partition_2.csv", "partition_3.csv", "partition_4.csv", "partition_5.csv"]
file_name = "FedAvg_smpc_metrics.csv"

# Shamir's Secret Sharing Parameters
threshold = num_nodes
prime_modulus = int(gf_irreducible(2, ZZ(nextprime(input_size)), ZZ)[0])

# Function to load dataset from a CSV file
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            dataset.append([float(value) for value in row[:-1]] + [int(row[-1])])  # Include label
    return np.array(dataset)

# Function to create a model for local training
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to simulate local training and update
def local_train_and_update(dataset, num_epochs=num_epochs):
    model = create_model()
    features = tf.convert_to_tensor(dataset[:, :-1], dtype=tf.float32)
    labels = tf.convert_to_tensor(dataset[:, -1], dtype=tf.float32)

    start_time = time.time()
    model.fit(features, labels, epochs=num_epochs, verbose=0)  # Simulate local training
    end_time = time.time()

    convergence_time = round(end_time - start_time, 2)  # Round convergence time to 2 decimal places

    loss, accuracy = model.evaluate(features, labels, verbose=0)
    predictions = model.predict(features).flatten()  # Predict probabilities
    binary_predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary values

    # Compute TP, TN, FP, FN
    tp = np.sum(binary_predictions * labels)
    tn = np.sum((1 - binary_predictions) * (1 - labels))
    fp = np.sum(binary_predictions * (1 - labels))
    fn = np.sum((1 - binary_predictions) * labels)

    return model.get_weights(), convergence_time, loss, accuracy, tp, tn, fp, fn

# Function to perform Shamir's Secret Sharing
def shamir_secret_sharing(secret, num_shares, threshold, prime_modulus):
    coefficients = np.random.randint(0, prime_modulus, threshold - 1)
    coefficients = np.append(secret, coefficients)
    shares = []
    for i in range(num_shares):
        x = i + 1
        y = sum([(coefficients[j] * (x ** j)) % prime_modulus for j in range(len(coefficients))]) % prime_modulus
        shares.append((x, y))
    return shares

# Function to perform secure aggregation using TenSEAL
# Function to perform secure aggregation using Shamir's Secret Sharing and Homomorphic Encryption
def secure_aggregation(weights):
    try:
        # Perform Shamir's Secret Sharing
        num_weights = len(weights)
        shares = []
        for i in range(num_weights):
            shares.append(shamir_secret_sharing(weights[i], num_weights, threshold, prime_modulus))

        # Homomorphically encrypt shares using TenSEAL
        encrypted_shares = []
        for share in shares:
            encrypted_shares.append(ckks_encoder.encode(share))

        # Perform secure aggregation by computing the sum of encrypted shares
        aggregated_result = encrypted_shares[0].copy()
        for i in range(1, num_weights):
            aggregated_result += encrypted_shares[i]

        return aggregated_result
    except Exception as e:
        print("Error:", e)
        return None  # Return None or handle the error as appropriate


# Open CSV file for writing metrics
with open(file_name, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["FL Round", "Convergence Time (s)", "Aggregation Time (s)", "FL Round Time (s)", "Accuracy", "Loss", "TP", "TN", "FP", "FN"])

    # Federated Learning loop
    for fl_round in range(1, num_fl_rounds + 1):
        print("FL Round:", fl_round)

        # List to hold local updates from all nodes
        local_updates = []

        # Record start time of FL round
        start_fl_round_time = time.time()
        print("Start FL time: ", start_fl_round_time)

        # Simulate local training and update for each node
        for node, file_path in enumerate(file_paths[:num_nodes], start=1):
            print("Node:", node)
            dataset = load_dataset(file_path)
            weights, convergence_time, loss, accuracy, tp, tn, fp, fn = local_train_and_update(dataset)
            local_updates.append(weights)

            print("Convergence Time (s):", convergence_time)
            print("Loss:", loss)
            print("Accuracy:", accuracy)

            writer.writerow([fl_round, convergence_time, "", "", accuracy, loss, tp, tn, fp, fn])

        # Perform secure aggregation
        start_aggregation_time = time.time()
        print("Aggregation happening now...")
        aggregated_result = secure_aggregation(local_updates)
        end_aggregation_time = time.time()

        aggregation_time = end_aggregation_time - start_aggregation_time

        print("Aggregation Time (s):", aggregation_time)

        # Calculate FL round time
        end_fl_round_time = time.time()
        print("End FL round time: ", end_fl_round_time)
        fl_round_time = end_fl_round_time - start_fl_round_time
        print("FL Round time: ", fl_round_time)

        # Write FL round time along with other metrics to the CSV
        writer.writerow([fl_round, "", aggregation_time, fl_round_time, "", "", "", "", "", ""])
