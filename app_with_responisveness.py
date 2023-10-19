import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Function to extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=1024))  # Use a smaller n_fft value
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function to load data and extract features for each sound file
def load_data(data_path, test_size=0.2):
    x, y = [], []
    for file in glob.glob(os.path.join(data_path, "Actor_*", "*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    if not x:
        raise ValueError("No sound files found in the specified path.")

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Function to print predicted emotions as a table
def print_emotion_table(y_test, y_pred):
    emotion_results_df = pd.DataFrame({'Audio File': y_test, 'Actual Emotion': y_test, 'Predicted Emotion': y_pred})
    print("\nActual and Predicted Emotions Table:")
    print(emotion_results_df)

# Function to print predicted responsiveness as a table
def print_responsiveness_table(predicted_responsiveness):
    responsiveness_results_df = pd.DataFrame({'Predicted Responsiveness': predicted_responsiveness})
    print("\nPredicted Responsiveness Table:")
    print(responsiveness_results_df)

# Function to plot smooth line chart for emotions and responsiveness
def plot_results(emotion_pred, responsiveness_pred, time_axis):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, emotion_pred, label='Predicted Emotion', marker='o', linestyle='-', color='blue')
    plt.title('Emotion Prediction Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Emotion')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, responsiveness_pred, label='Predicted Responsiveness', marker='o', linestyle='-', color='green')
    plt.title('Responsiveness Prediction Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Responsiveness')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Replace the path with your dataset path
dataset_path = r"C:\Users\JEEVAN\OneDrive\Desktop\Assignment\speech-emotion-recognition-ravdess-data"

try:
    # Split the dataset
    x_train, x_test, y_train, y_test = load_data(dataset_path, test_size=0.25)

    # Get the shape of the training and testing datasets
    print((x_train.shape[0], x_test.shape[0]))

    # Get the number of features extracted
    print(f'Features extracted: {x_train.shape[1]}')

    # Initialize the Multi-Layer Perceptron Classifier
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

    # Train the model
    model.fit(x_train, y_train)
    joblib.dump(model, 'trained_model.pkl')

    # Predict for the test set
    y_pred = model.predict(x_test)

    # Print the actual and predicted emotion tables
    print_emotion_table(y_test, y_pred)

    # Print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Replace the path with the path to your single sample
    file_path1 = 'happy.wav'
    single_sample_feature = extract_feature(file_path1, mfcc=True, chroma=True, mel=True)

    # Reshape the single sample feature to have the same shape as the training data
    single_sample_feature = single_sample_feature.reshape(1, -1)

    # Make the prediction for the single sample
    predicted_emotion_single = model.predict(single_sample_feature)[0]
    print(f'\nPredicted Emotion for {file_path1}: {predicted_emotion_single}')

    # Create a DataFrame for predicted responsiveness for the single sample
    sample_responsiveness_pred = model.predict_proba(single_sample_feature)[0]
    responsiveness_results_df_single = pd.DataFrame({'Predicted Responsiveness': sample_responsiveness_pred})

    # Print predicted responsiveness table for the single sample
    print("\nPredicted Responsiveness Table for the Single Sample:")
    print(responsiveness_results_df_single)

    # Example usage:
    # Replace the following with your actual predicted data
    sample_emotion_pred = np.random.choice(['happy', 'sad'], size=len(y_test))
    sample_responsiveness_pred = np.random.rand(len(y_test))
    time_axis = np.arange(len(y_test))

    # Plotting the example
    plot_results(sample_emotion_pred, sample_responsiveness_pred, time_axis)

except Exception as e:
    print(f"Error: {e}")
