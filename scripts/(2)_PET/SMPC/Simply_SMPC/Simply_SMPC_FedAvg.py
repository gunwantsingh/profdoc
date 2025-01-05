import csv
import time
import tensorflow as tf
import numpy as np

# Global model parameters
input_size = 24  # Number of features in the dataset, adjust accordingly
output_size = 1
learning_rate = 0.1
num_epochs = 3
num_nodes = 5  # Number of nodes in the federated learning setup
num_fl_rounds = 2  # Number of Federated Learning rounds
file_paths = ["partition_1.csv", "partition_2.csv", "partition_3.csv", "partition_4.csv", "partition_5.csv"]
file_name = "FedAvg_metrics.csv"

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

# Open CSV file for writing metrics
with open(file_name, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["FL Round", "Convergence Time (s)", "Aggregation Time (s)", "Accuracy", "Loss", "TP", "TN", "FP", "FN"])

    # Federated Learning loop
    for fl_round in range(1, num_fl_rounds + 1):
        print("FL Round:", fl_round)

        # List to hold local updates from all nodes
        local_updates = []

        # Record start time of FL round
        start_fl_round_time = time.time()

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

        # Perform model aggregation (FedAvg)
        aggregated_result = np.mean(np.array(local_updates), axis=0)  # Compute mean of local updates

        # Record aggregation time
        end_fl_round_time = time.time()
        aggregation_time = round(end_fl_round_time - start_fl_round_time, 2)

        # Update global model with aggregated result
        # global_model.set_weights(aggregated_result)  # Uncomment if using a global model

        # Write FL round and aggregation time metrics
        writer.writerow([fl_round, "", aggregation_time, "", "", "", "", "", ""])

print("Federated learning completed.")




# import csv
# import time
# import tensorflow as tf
# import numpy as np

# # Global model parameters
# input_size = 24  # Number of features in the dataset, adjust accordingly
# output_size = 1
# learning_rate = 0.1
# num_epochs = 30
# num_nodes = 5  # Number of nodes in the federated learning setup
# num_fl_rounds = 10  # Number of Federated Learning rounds
# file_paths = ["partition_1.csv", "partition_2.csv", "partition_3.csv", "partition_4.csv", "partition_5.csv"]
# file_name = "FedAvg_smpc_simply_metrics.csv"

# # Function to load dataset from a CSV file
# def load_dataset(file_path):
#     dataset = []
#     with open(file_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header
#         for row in reader:
#             dataset.append([float(value) for value in row[:-1]] + [int(row[-1])])  # Include label
#     return np.array(dataset)

# # Function to create a model for local training
# def create_model():
#     input_main = tf.keras.layers.Input(shape=(input_size,))
#     output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(input_main)

#     model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)
#     model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
#                   loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Function to simulate local training and update
# def local_train_and_update(dataset, num_epochs=num_epochs):
#     model = create_model()
#     features = tf.convert_to_tensor(dataset[:, :-1], dtype=tf.float32)
#     labels = tf.convert_to_tensor(dataset[:, -1], dtype=tf.float32)

#     start_time = time.time()
#     model.fit(features, labels, epochs=num_epochs, verbose=0)  # Simulate local training
#     end_time = time.time()

#     convergence_time = round(end_time - start_time, 2)  # Round convergence time to 2 decimal places

#     loss, accuracy = model.evaluate(features, labels, verbose=0)
#     predictions = model.predict(features).flatten()  # Predict probabilities
#     binary_predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary values

#     # Compute TP, TN, FP, FN
#     tp = np.sum(binary_predictions * labels)
#     tn = np.sum((1 - binary_predictions) * (1 - labels))
#     fp = np.sum(binary_predictions * (1 - labels))
#     fn = np.sum((1 - binary_predictions) * labels)

#     return model.get_weights(), convergence_time, loss, accuracy, tp, tn, fp, fn

# # Open CSV file for writing metrics
# with open(file_name, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["FL Round", "Convergence Time (s)", "Accuracy", "Loss", "TP", "TN", "FP", "FN"])

#     # Federated Learning loop
#     for fl_round in range(1, num_fl_rounds + 1):
#         print("FL Round:", fl_round)

#         # List to hold local updates from all nodes
#         local_updates = []

#         # Simulate local training and update for each node
#         for node, file_path in enumerate(file_paths[:num_nodes], start=1):
#             print("Node:", node)
#             dataset = load_dataset(file_path)
#             weights, convergence_time, loss, accuracy, tp, tn, fp, fn = local_train_and_update(dataset)
#             local_updates.append(weights)

#             print("Convergence Time (s):", convergence_time)
#             print("Loss:", loss)
#             print("Accuracy:", accuracy)

#             writer.writerow([fl_round, convergence_time, accuracy, loss, tp, tn, fp, fn])
