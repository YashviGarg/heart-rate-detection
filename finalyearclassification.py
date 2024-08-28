# -*- coding: utf-8 -*-
"""HeartClassification.ipynb"""

# Load an audio signal and plot its waveform
import librosa
import matplotlib.pyplot as plt
signal, sr = librosa.load('/project/testCases/TestCase40.wav', sr=44100)
plt.figure(figsize=(4, 4))
plt.plot(signal)


# Play an audio file in the notebook
import IPython.display as ipd
ipd.Audio('/project/testCases/TestCase1.wav')

# Set the path for the audio dataset and load the metadata CSV file
import pandas as pd
import os
import librosa
audio_dataset_path='project/testCases/'
metadata=pd.read_csv('project/final_dataset.csv')
metadata.head()

import numpy as np
import random

# Function to add random noise to the audio data
def manipulate0(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

# Function to shift the audio data in time
def manipulate1(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

# Function to change the pitch of the audio data
def manipulate2(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# Function to change the speed of the audio data
def manipulate3(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

import os
import librosa
from librosa.core.convert import mel_frequencies
import numpy as np
from tqdm import tqdm

# Keyword arguments for Mel-spectrogram calculation
kwargs = {"htk": True, "norm": "slaney"}

# Function to randomly apply one of the data augmentation techniques
def switch_function(signal,sr):
    n = random.randint(0, 3)  # Randomly select one of the four manipulations
    M0 = [.0050, .0015, .0025, .0060, .0040]  # Noise factors
    M1 = [0.1, 0.2, 0.15, 0.25, 0.3]  # Shift factors
    M2 = [0.5, 1, 1.5, 2, 2.5]  # Pitch factors
    M3 = [0.5, 0.75, 1, 1.15, 1.25]  # Speed factors
    signal={
        0: manipulate0(signal,random.choice(M0)),
        1: manipulate1(signal,sr,random.choice(M1),'both'),
        2: manipulate2(signal,sr,random.choice(M2)),
        3: manipulate3(signal,random.choice(M3))
    }
    return signal.get(n,"Invalid")

# Function to extract features from the audio signal
def features_extractor(sig,sr):
    preemphasized_signal = librosa.effects.preemphasis(sig, coef=0.95)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=preemphasized_signal,
        sr=44100,
        n_mels=42,
        n_fft=512,
        hop_length=100,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        **kwargs
    )

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=12, dct_type=2)
    mfcc_scaled_features = np.mean(mfcc.T, axis=0)

    return [mfcc_scaled_features, mel_spectrogram]



# Lists to store extracted features and spectrogram data
extracted_features = []
spectogram_data = []

# Loop through each row in the metadata, extracting features from the corresponding audio files
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row["file_name"])) # Get file path
    gender_labels = row["gender"]
    age_group_labels = row["age_group"]
    state_labels = row["state"]
    bpm_labels = row["bpm"]
    heart_rate_labels = row["heart_rate"]
    
    # Load the audio file
    signal, sr = librosa.load(file_name, sr=44100)
    # Extract features from the original audio
    data1 = features_extractor(signal,sr)
    # Apply data augmentation and extract features again
    signal1 = switch_function(signal,sr)
    data2 = features_extractor(signal1,sr)
    # Apply another augmentation and extract features again
    signal2 = switch_function(signal,sr)
    data3 = features_extractor(signal2,sr)
    # Store the spectrogram data
    spectogram_data.append(data1[1])
    # Append the extracted features and corresponding labels
    extracted_features.append(
        [data1[0], gender_labels, age_group_labels, state_labels, bpm_labels, heart_rate_labels]
    )
    extracted_features.append(
        [data2[0], gender_labels, age_group_labels, state_labels, heart_rate_labels]
    )
    extracted_features.append(
        [data3[0], gender_labels, age_group_labels, state_labels, heart_rate_labels]
    )

# Convert categorical variables (e.g., gender, state, age group) into one-hot encoded vectors
one_hot = pd.get_dummies(metadata["gender"])
metadata = metadata.drop("gender", axis=1)
metadata = metadata.join(one_hot)

# Clean and encode the state labels
newstates= []
states = metadata["state"].to_list()
for state in states:
    newstates.append(
        state.lower().strip()
    )
metadata = metadata.drop("state", axis=1)
metadata['state'] = newstates

one_hot = pd.get_dummies(metadata["state"])
metadata = metadata.drop("state", axis=1)
metadata = metadata.join(one_hot)

# Encode the age group labels
one_hot = pd.get_dummies(metadata["age_group"])
metadata = metadata.drop("age_group", axis=1)
metadata = metadata.join(one_hot)

heart_rates = metadata["heart_rate"].to_list()

metadata = metadata.drop("heart_rate", axis=1)
metadata = metadata.drop("file_name", axis=1)

# Prepare the target variable (bpm)
y = metadata["bpm"]
metadata = metadata.drop("bpm",axis=1)
metadata = metadata.join(y)


# Prepare the feature matrix
remaining_features = metadata.values
result = []

# Combine extracted features with the remaining metadata features
for i in range(len(extracted_features)):
    features = extracted_features[i][0].tolist()
    new = []

    new += extracted_features[i][0].tolist()
    new += remaining_features[i%104].tolist()
    new.append(heart_rates[i%104])

    result.append(new)

# Define column names for the final dataset
names = []
for i in range(1,13):
    names.append(f"mfcc_{i}")
s = "F 	M 	after_workout 	happy 	neutral 	relaxed 	stressed 	tired 	20-29 	30-39 	40-49 	50-59 	60-69 	bpm"

names = names + s.strip().split()
names.append("heart_rate")

# Save the final dataset to a CSV file
pd.DataFrame(result, columns=names).to_csv("save.csv", index=False)

# Plot the spectrogram of the second audio sample in the spectrogram data
log_spect = librosa.power_to_db(spectogram_data[1])
plt.figure(figsize=(6,4))
librosa.display.specshow(log_spect,x_axis = "time",sr=44100,hop_length=100)
plt.xlabel("time")
plt.ylabel("frequency")
plt.colorbar()
plt.show()

# Load the extracted features into a DataFrame and display the first few rows
extracted_features_df = pd.read_csv("save.csv")
extracted_features_df.head()



### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df.iloc[:, :-1].values.tolist())
y=np.array(extracted_features_df.iloc[:, -1].values.tolist())

X.shape

### Label Encoding
y=np.array(pd.get_dummies(y))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

y.shape

### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Standardize the feature matrices
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(50,input_shape=(26,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
###second layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
###third layer
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Display the model's architecture
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# Train the model with the training data
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 50
num_batch_size = 5

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)

start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


# Calculate the duration of the training process
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluate the model on the test data
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Generate and display the confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
multilabel_confusion_matrix(y_test, y_pred)

# Calculate and display accuracy, precision, recall, and F1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

# Generate and display a detailed classification report
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2']))

