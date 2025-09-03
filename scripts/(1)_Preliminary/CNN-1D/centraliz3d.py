import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from keras.models import Sequential



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data
# Replace this with your intrusion detection dataset
num_samples = 1000
sequence_length = 100
num_classes = 2

X = np.random.randn(num_samples, sequence_length, 1)
y = np.random.randint(0, num_classes, size=num_samples)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple 1D CNN model
model = keras.Sequential([
    keras.Conv1D(64, 3, activation='relu', input_shape=(sequence_length, 1)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
# start_time = time.time()
s = time.time()
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
e = time.time()
# end_time = time.time()

# Calculate convergence time
# convergence_time = end_time - start_time
convergence_time = e - s

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)

# Calculate confusion matrix to get true positives and false positives
confusion = confusion_matrix(y_test, y_pred_classes)
true_positives = confusion[1, 1]
false_positives = confusion[0, 1]

# Print results
print(f"Convergence Time: {convergence_time} seconds")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
