import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Define dataset path (update to your local directory containing RAVDESS .wav files)
DATASET_PATH = "./all voices"  # Updated to match your folder structure

# Emotion map for RAVDESS
emotion_map = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '02': 'calm', '06': 'fearful', '07': 'surprise', '08': 'disgust'}

def get_emotion(filename):
    try:
        parts = filename.split('-')
        return emotion_map.get(parts[2])
    except IndexError:
        print(f"Skipping invalid file: {filename}")
        return None

def extract_features(file_path, max_len=130):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    delta_mfcc = librosa.feature.delta(mfcc.T).T
    features = np.concatenate([mfcc, delta_mfcc], axis=1)
    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)), mode='constant')
    else:
        features = features[:max_len]
    return features

# Load and preprocess data
X, y = [], []
for root, _, filenames in os.walk(DATASET_PATH):
    for file in filenames:
        if file.endswith('.wav'):
            emotion = get_emotion(file)
            if emotion:
                X.append(extract_features(os.path.join(root, file)))
                y.append(emotion)

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y = to_categorical(y_encoded)

# Save for reuse
np.save("X.npy", X)
np.save("y.npy", y)
np.save("classes.npy", le.classes_)

# Load saved features
X = np.load("X.npy")
y = np.load("y.npy")
classes = np.load("classes.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping, lr_scheduler])

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))

# Save model
model.save("emotion_rnn.h5")
print("Model saved as emotion_rnn.h5")

import os
import numpy as np
import librosa
from keras.models import load_model

from pydub import AudioSegment
import time

# Define extract_features
def extract_features(file_path, max_len=130):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    delta_mfcc = librosa.feature.delta(mfcc.T).T
    features = np.concatenate([mfcc, delta_mfcc], axis=1)
    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)), mode='constant')
    else:
        features = features[:max_len]
    return features

# Check if required files exist
if not os.path.exists("emotion_rnn.h5"):
    raise FileNotFoundError("Model file 'emotion_rnn.h5' not found. Please run train.py to train and save the model.")
if not os.path.exists("classes.npy"):
    raise FileNotFoundError("Classes file 'classes.npy' not found. Please run train.py to generate data files.")

# Timing: Load model
start = time.time()
model = load_model("emotion_rnn.h5")
print(f"Model loading time: {time.time() - start:.2f} seconds")

# Re-compile model to avoid metrics warning
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(f"Model compilation time: {time.time() - start:.2f} seconds")

# Timing: Load classes
start = time.time()
classes = np.load("classes.npy")
print(f"Classes loading time: {time.time() - start:.2f} seconds")

# Get audio file path from user
file_path = input("Enter the path to your audio file (e.g., ./test_audio/Recording (3).m4a): ")

# Supported formats
supported_formats = ['.wav', '.m4a', '.mp3', '.ogg', '.flac', '.aac']

# Check if file exists and format is supported
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Audio file not found: {file_path}")
if not any(file_path.lower().endswith(fmt) for fmt in supported_formats):
    raise ValueError(f"File must be one of: {', '.join(supported_formats)}")

# Timing: Convert non-.wav files
start = time.time()
if not file_path.lower().endswith('.wav'):
    print(f"Converting {file_path} to .wav...")
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.splitext(file_path)[0] + '.wav'
        audio.export(wav_path, format="wav")
        file_path = wav_path
    except Exception as e:
        raise ValueError(f"Failed to convert {file_path} to .wav: {str(e)}")
print(f"File conversion time: {time.time() - start:.2f} seconds")

# Timing: Extract features
start = time.time()
try:
    mfcc = extract_features(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
except Exception as e:
    raise ValueError(f"Failed to extract features: {str(e)}")
print(f"Feature extraction time: {time.time() - start:.2f} seconds")

# Timing: Predict
start = time.time()
pred = model.predict(mfcc)
predicted_emotion = classes[np.argmax(pred)]
print(f"Prediction time: {time.time() - start:.2f} seconds")

print(f"Predicted Emotion: {predicted_emotion}")