import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import queue
import tensorflow as tf

# Function to load dataset
def load_dataset(partition):
    try:
        return pd.read_csv(f"partition_{partition}.csv")
    except FileNotFoundError:
        print(f"Error: partition_{partition}.csv not found.")
        return None

# Function to create Keras model for binary classification
def create_model(input_shape=(24,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
    return model

# Function to calculate gradient norms
def calculate_gradient_norms(model, X_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(X_train, training=True)
        loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
    avg_gradient_norm = np.mean(gradient_norms)
    return avg_gradient_norm

# Function to ensure target shape matches model output shape
def adjust_target_shape(target, model_output_shape):
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    return target

# Function for local training on a node
def local_training(node_id, X_train, y_train, X_test, y_test, local_model, update_queues, max_fl_rounds, num_epochs):
    fl_round = 0
    metrics = []

    while fl_round < max_fl_rounds:
        round_start_time = time.time()
        loss_history = []
        convergence_time = None

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Adjust the shape of the target labels
            y_train = adjust_target_shape(y_train, local_model.output_shape)
            y_test = adjust_target_shape(y_test, local_model.output_shape)

            history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
            epoch_end_time = time.time()

            current_loss = history.history['loss'][0]
            loss_history.append(current_loss)

            # Calculate validation accuracy and loss using the test dataset
            validation_loss, validation_accuracy = local_model.evaluate(X_test, y_test, verbose=0)

            y_pred = local_model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()

            avg_gradient_norm = calculate_gradient_norms(local_model, X_train, y_train)

            # Capture node metrics
            metrics.append({
                'FL_Round': fl_round + 1,
                'Epoch': epoch + 1,
                'Node_ID': node_id,
                'Accuracy': history.history['accuracy'][0],
                'Loss': current_loss,
                'Validation_Accuracy': validation_accuracy,
                'Validation_Loss': validation_loss,
                'FP': fp,
                'TP': tp,
                'FN': fn,
                'TN': tn,
                'Gradient_Norm': avg_gradient_norm,
                'Epoch_Time': epoch_end_time - epoch_start_time,
                'Aggregation_Time': 0,  # This will be updated after aggregation
                'FL_Round_Time': 0,  # This will be updated after round completion
                'Convergence_Time': 0  # This will be updated after convergence
            })

            print(f"Node {node_id} (FL Round {fl_round + 1}, Epoch {epoch + 1}): Training completed with accuracy: {history.history['accuracy'][0]:.4f}, loss: {current_loss:.4f}, validation accuracy: {validation_accuracy:.4f}, validation loss: {validation_loss:.4f}, gradient norm: {avg_gradient_norm:.4f}, time: {epoch_end_time - epoch_start_time:.2f}s")

            # Check for convergence by looking at the last 5 loss values
            if len(loss_history) >= 5:
                recent_losses = loss_history[-5:]
                if max(recent_losses) - min(recent_losses) < 1e-4:  # Convergence threshold
                    convergence_time = time.time() - round_start_time
                    print(f"Node {node_id}: Convergence reached at Epoch {epoch + 1} of FL Round {fl_round + 1}. Convergence Time: {convergence_time:.2f}s")
                    break  # Stop the current FL round

            # Send local model update to other nodes via queues
            for neighbor_id, queue in update_queues.items():
                if neighbor_id != node_id:  # Skip sending update to self
                    queue.put((local_model.get_weights(), node_id))

            # Aggregation step
            aggregation_start_time = time.time()
            update_model(local_model, update_queues[node_id])
            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            # Update metrics with aggregation time
            metrics[-1]['Aggregation_Time'] = aggregation_time

        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        metrics[-1]['FL_Round_Time'] = round_time

        if convergence_time is not None:
            for metric in metrics:
                metric['Convergence_Time'] = convergence_time
        else:
            for metric in metrics:
                metric['Convergence_Time'] = round_time  # If no convergence, use round time as a fallback

        fl_round += 1

    return metrics

# Function to update models based on received weights and capture aggregation time
def update_model(local_model, update_queue):
    local_weights = []
    try:
        while True:
            weights, sender_id = update_queue.get_nowait()
            local_weights.append(weights)
    except queue.Empty:
        pass

    if local_weights:
        new_weights = []
        num_layers = len(local_model.get_weights())

        for layer_idx in range(num_layers):
            layer_weights = np.mean([weights[layer_idx] for weights in local_weights], axis=0)
            new_weights.append(layer_weights)

        local_model.set_weights(new_weights)

# Function to start the asynchronous P2P federated learning
def start_federated_learning(nodes_data, max_fl_rounds, num_epochs):
    local_models = [create_model(input_shape=nodes_data[0][0].shape[1:]) for _ in range(len(nodes_data))]
    update_queues = {node_id: queue.Queue() for node_id in range(len(nodes_data))}
    all_metrics = []

    for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
        node_metrics = local_training(node_id, X_train, y_train, X_test, y_test, local_models[node_id], update_queues, max_fl_rounds, num_epochs)
        all_metrics.extend(node_metrics)

    # Convert metrics to DataFrame and save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('__M3_Final_Asynchronous_P2P_Federated_Learning_Metrics.csv', index=False)
    print("Metrics saved to 'Final_Asynchronous_P2P_Federated_Learning_Metrics.csv'.")

# Load datasets for each node
nodes_data = []
for partition in range(1, 6):
    data = load_dataset(partition)
    if data is not None and 'label' in data.columns:
        X = data.drop(columns=['label']).values
        y = data['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        nodes_data.append((X_train, y_train, X_test, y_test))
    else:
        print(f"Skipping partition_{partition} due to missing data or missing 'label' column.")

# Start federated learning
max_fl_rounds = 15  # Define the maximum number of FL rounds
num_epochs = 40  # Define the number of epochs per FL round
start_federated_learning(nodes_data, max_fl_rounds, num_epochs)



# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, Input
# from keras.optimizers import SGD
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import time
# import queue
# import tensorflow as tf

# # Function to load dataset
# def load_dataset(partition):
#     try:
#         return pd.read_csv(f"partition_{partition}.csv")
#     except FileNotFoundError:
#         print(f"Error: partition_{partition}.csv not found.")
#         return None

# # Function to create Keras model for binary classification
# def create_model(input_shape=(24,)):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
#     return model

# # Function to calculate gradient norms
# def calculate_gradient_norms(model, X_train, y_train):
#     with tf.GradientTape() as tape:
#         predictions = model(X_train, training=True)
#         loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
#     avg_gradient_norm = np.mean(gradient_norms)
#     return avg_gradient_norm

# # Function to ensure target shape matches model output shape
# def adjust_target_shape(target, model_output_shape):
#     if target.ndim == 1:
#         target = target.reshape(-1, 1)
#     return target

# # Function for local training on a node
# def local_training(node_id, X_train, y_train, X_test, y_test, local_model, update_queues, max_fl_rounds, num_epochs):
#     fl_round = 0
#     metrics = []

#     while fl_round < max_fl_rounds:
#         round_start_time = time.time()
#         loss_history = []

#         for epoch in range(num_epochs):
#             epoch_start_time = time.time()

#             # Adjust the shape of the target labels
#             y_train = adjust_target_shape(y_train, local_model.output_shape)
#             y_test = adjust_target_shape(y_test, local_model.output_shape)

#             history = local_model.fit(X_train, y_train, epochs=1, verbose=0)  # Train for 1 epoch
#             epoch_end_time = time.time()

#             current_loss = history.history['loss'][0]
#             loss_history.append(current_loss)

#             # Calculate validation accuracy and loss using the test dataset
#             validation_loss, validation_accuracy = local_model.evaluate(X_test, y_test, verbose=0)

#             y_pred = local_model.predict(X_test)
#             y_pred_binary = (y_pred > 0.5).astype(int)
#             cm = confusion_matrix(y_test, y_pred_binary)
#             tn, fp, fn, tp = cm.ravel()

#             avg_gradient_norm = calculate_gradient_norms(local_model, X_train, y_train)

#             # Capture node metrics
#             metrics.append({
#                 'FL_Round': fl_round + 1,
#                 'Epoch': epoch + 1,
#                 'Node_ID': node_id,
#                 'Accuracy': history.history['accuracy'][0],
#                 'Loss': current_loss,
#                 'Validation_Accuracy': validation_accuracy,
#                 'Validation_Loss': validation_loss,
#                 'FP': fp,
#                 'TP': tp,
#                 'FN': fn,
#                 'TN': tn,
#                 'Gradient_Norm': avg_gradient_norm,
#                 'Epoch_Time': epoch_end_time - epoch_start_time,
#                 'Aggregation_Time': 0,  # This will be updated after aggregation
#                 'FL_Round_Time': 0,  # This will be updated after round completion
#                 'Convergence_Time': 0  # This will be updated after final round
#             })

#             print(f"Node {node_id} (FL Round {fl_round + 1}, Epoch {epoch + 1}): Training completed with accuracy: {history.history['accuracy'][0]:.4f}, loss: {current_loss:.4f}, validation accuracy: {validation_accuracy:.4f}, validation loss: {validation_loss:.4f}, gradient norm: {avg_gradient_norm:.4f}, time: {epoch_end_time - epoch_start_time:.2f}s")

#             # Check for convergence by looking at the last 5 loss values
#             if len(loss_history) >= 5:
#                 recent_losses = loss_history[-5:]
#                 if max(recent_losses) - min(recent_losses) < 1e-4:  # Convergence threshold
#                     print(f"Node {node_id}: Convergence reached at Epoch {epoch + 1} of FL Round {fl_round + 1}.")
#                     break  # Stop the current FL round

#             # Send local model update to other nodes via queues
#             for neighbor_id, queue in update_queues.items():
#                 if neighbor_id != node_id:  # Skip sending update to self
#                     queue.put((local_model.get_weights(), node_id))

#             # Aggregation step
#             aggregation_start_time = time.time()
#             update_model(local_model, update_queues[node_id])
#             aggregation_end_time = time.time()
#             aggregation_time = aggregation_end_time - aggregation_start_time

#             # Update metrics with aggregation time
#             metrics[-1]['Aggregation_Time'] = aggregation_time

#         round_end_time = time.time()
#         round_time = round_end_time - round_start_time
#         metrics[-1]['FL_Round_Time'] = round_time

#         fl_round += 1

#     # Calculate and store final convergence time
#     total_convergence_time = time.time() - round_start_time
#     for metric in metrics:
#         metric['Convergence_Time'] = total_convergence_time

#     return metrics

# # Function to update models based on received weights and capture aggregation time
# def update_model(local_model, update_queue):
#     local_weights = []
#     try:
#         while True:
#             weights, sender_id = update_queue.get_nowait()
#             local_weights.append(weights)
#     except queue.Empty:
#         pass

#     if local_weights:
#         new_weights = []
#         num_layers = len(local_model.get_weights())

#         for layer_idx in range(num_layers):
#             layer_weights = np.mean([weights[layer_idx] for weights in local_weights], axis=0)
#             new_weights.append(layer_weights)

#         local_model.set_weights(new_weights)

# # Function to start the asynchronous P2P federated learning
# def start_federated_learning(nodes_data, max_fl_rounds, num_epochs):
#     local_models = [create_model(input_shape=nodes_data[0][0].shape[1:]) for _ in range(len(nodes_data))]
#     update_queues = {node_id: queue.Queue() for node_id in range(len(nodes_data))}
#     all_metrics = []

#     for node_id, (X_train, y_train, X_test, y_test) in enumerate(nodes_data):
#         node_metrics = local_training(node_id, X_train, y_train, X_test, y_test, local_models[node_id], update_queues, max_fl_rounds, num_epochs)
#         all_metrics.extend(node_metrics)

#     # Convert metrics to DataFrame and save
#     metrics_df = pd.DataFrame(all_metrics)
#     metrics_df.to_csv('Final_Asynchronous_P2P_Federated_Learning_Metrics.csv', index=False)
#     print("Metrics saved to 'Asynchronous_P2P_Federated_Learning_Metrics.csv'.")

# # Load datasets for each node
# nodes_data = []
# for partition in range(1, 6):
#     data = load_dataset(partition)
#     if data is not None and 'label' in data.columns:
#         X = data.drop(columns=['label']).values
#         y = data['label'].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         nodes_data.append((X_train, y_train, X_test, y_test))
#     else:
#         print(f"Skipping partition_{partition} due to missing data or missing 'label' column.")

# # Start federated learning
# max_fl_rounds = 15  # Define the maximum number of FL rounds
# num_epochs = 40  # Define the number of epochs per FL round
# start_federated_learning(nodes_data, max_fl_rounds, num_epochs)
