import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Set the path for audio files
path = "/Users/Kermi/Downloads/ALL"
audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".wav")]

# Labels and Features Extraction
def extract_labels_and_features(audio_files):
    labels = []
    features = []
    for file in audio_files:
        # Extract the emotion label from the file name
        file_name = file.split("/")[-1].split("_")[-1]
        if file_name.startswith("n"):
            labels.append("neutral")
        elif file_name.startswith("h"):
            labels.append("happy")
        elif file_name.startswith("d"):
            labels.append("disgust")
        elif file_name.startswith("a"):
            labels.append("angry")
        elif file_name.startswith("c"):
            labels.append("calm")
        elif file_name.startswith("f"):
            labels.append("fearful")
        elif file_name.startswith("s"):
            if file_name[1] == "a":
                labels.append("sad")
            elif file_name[1] == "u":
                labels.append("surprise")
        else:
            continue

        # Extract features
        y, sr = librosa.load(file, duration=2.5, offset=0.6)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
        features.append(np.mean(mfcc.T, axis=0))
    return np.array(features), np.array(labels)

# Extract features and labels
print("Extracting features and labels...")
features, labels = extract_labels_and_features(audio_files)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)  # One-hot encode labels

# Save the label encoder for deployment
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved as label_encoder.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for deployment
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl")

# Reshape features for RNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the RNN model
print("Building the RNN model...")
model = Sequential()
model.add(SimpleRNN(256, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
batch_size = 64
epochs = 50
print("Training the model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=True)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save("audio_emotion_rnn.h5")
print("Model saved as audio_emotion_rnn.h5")

# Predict and evaluate on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion matrix and metrics
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average="weighted")
recall = recall_score(y_true_labels, y_pred_labels, average="weighted")
f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")

print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
