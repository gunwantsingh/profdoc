import numpy as np
import pandas as pd
import tensorflow as tf
import time

# Define a simple Keras model
def create_model():
    input_main = tf.keras.layers.Input(shape=(input_size,))
    hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_main)  # Add a hidden layer
    hidden_layer2 = tf.keras.layers.Dense(32, activation='relu')(hidden_layer1)  # Add another hidden layer
    output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')(hidden_layer2)

    model = tf.keras.models.Model(inputs=input_main, outputs=output_layer)

    # Use SGD optimizer for FedDP
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

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

def compute_privacy(train_data, num_examples, batch_size, noise_multiplier, epochs):
    try:
        # Placeholder for computing privacy
        epsilon = None
        return epsilon
    except Exception as e:
        print(f"An error occurred while computing privacy: {e}")
        return None

def feddp_aggregate(local_models):
    global_model_weights = global_model_updated.get_weights()
    num_layers = len(global_model_weights)

    for layer in range(num_layers):
        layer_weights = [model_weights[layer] for model_weights in local_models]
        aggregated_weights = np.median(layer_weights, axis=0)
        global_model_weights[layer] = aggregated_weights

    global_model_updated.set_weights(global_model_weights)

try:
    file_path = "../../BotNetIoT.csv"  # Update with the actual path to your dataset file
    loaded_dataset = load_dataset(file_path)

    if loaded_dataset is not None:
        train_size = int(0.8 * len(loaded_dataset))
        train_data, val_data = loaded_dataset[:train_size], loaded_dataset[train_size:]

        clients = 5
        local_models = []
        metrics = []

        for round_num in range(FL_rounds):
            round_metrics = []

            print(f"\nFederated Learning Round {round_num + 1}:")
            round_start_time = time.time()

            for i in range(clients):
                local_data = train_data.sample(frac=0.1)
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

            feddp_aggregate(local_models)

            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            epsilon = compute_privacy(train_data, len(train_data), batch_size=256, noise_multiplier=0.5, epochs=3)

            round_end_time = time.time()
            round_time = round_end_time - round_start_time

            print(f"  Aggregation Time for FL Round {round_num + 1}: {aggregation_time:.4f} seconds")
            print(f"  Round Time: {round_time:.4f} seconds")
            print(f"  Epsilon: {epsilon}")

            round_metrics.append({
                'Aggregation Time': aggregation_time,
                'Round Time': round_time,
                'Epsilon': epsilon
            })

            metrics.extend(round_metrics)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv('Simply_FedDP.csv', index=False)

except Exception as e:
    print(f"An error occurred: {e}")
