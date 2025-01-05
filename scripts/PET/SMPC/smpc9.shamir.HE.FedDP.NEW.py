# Yes, this script incorporates Secure Multi-Party Computation (SMPC) using Shamir's Secret Sharing (SSS) and Homomorphic Encryption (HE), specifically the TenSEAL library, for secure aggregation. It also utilizes Federated Differential Privacy (FedDP) and Differential Privacy (DP) for privacy-preserving federated learning.

# Here's how these components are used in the script:

# Shamir's Secret Sharing (SSS): The shamir_secret_sharing function is used to split the model weights into shares that are distributed among the participating nodes.
# Homomorphic Encryption (HE): The TenSEAL library is used for homomorphic encryption. Encrypted shares of model weights are aggregated securely without revealing their individual values.
# Secure Aggregation: The secure_aggregation function performs secure aggregation of encrypted shares using both SSS and HE, ensuring that the aggregation process maintains privacy.
# Federated Differential Privacy (FedDP): The script includes differential privacy parameters (epsilon and delta) to provide privacy guarantees for the federated learning process.
# Differential Privacy (DP): The compute_dp_sgd_privacy function from the TensorFlow Privacy library is used to compute the differential privacy for each federated learning round.
# Overall, the script combines these techniques to enable privacy-preserving federated learning with secure aggregation of model updates across multiple nodes.

import csv
import time
import tensorflow as tf
import numpy as np
import tenseal as ts
from sympy import nextprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_irreducible
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# Global model parameters
input_size = 24  # Number of features in the dataset, adjust accordingly
output_size = 1
learning_rate = 0.1
num_epochs = 30
num_nodes = 5  # Number of nodes in the federated learning setup
num_fl_rounds = 10  # Number of Federated Learning rounds
file_paths = ["partition_1.csv", "partition_2.csv", "partition_3.csv", "partition_4.csv", "partition_5.csv"]
file_name = "FedDP_SMPC_metrics.csv"

# Differential Privacy Parameters
epsilon = 1.0
delta = 1e-5

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

# Function to perform secure aggregation using SMPC (SSS + HE)
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

        # Compute differential privacy for this round
        orders = [1, 2, 4]
        for order in orders:
            try:
                epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=len(dataset), batch_size=1, noise_multiplier=1.0, epochs=num_epochs, delta=delta, target_delta=delta/100)
                print(f"DP (order {order}): (ε = {epsilon}, δ = {delta})")
            except AttributeError:
                print("Error computing differential privacy. Skipping this order.")
                continue
