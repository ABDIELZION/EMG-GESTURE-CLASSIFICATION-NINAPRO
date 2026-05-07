import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

file_paths = ['S1_E1_a1.mat']
file_paths = ['S1_E2_a1.mat']
file_paths = ['S1_E3_a1.mat']


def load_and_preprocess(files):
    all_emg = []
    all_labels = []

    for f in files:
        mat = scipy.io.loadmat(f)
        emg = mat['emg']

        label = mat.get('stimulus', mat.get('restimulus'))

        all_emg.append(emg)
        all_labels.append(label.flatten())

    return np.vstack(all_emg), np.concatenate(all_labels)


print("Status: Loading MATLAB files...")
X_raw, y_raw = load_and_preprocess(file_paths)


WINDOW_SIZE = 200
STEP = 100


def extract_features(data, labels):
    features = []
    targets = []
    for i in range(0, len(data) - WINDOW_SIZE, STEP):
        window = data[i: i + WINDOW_SIZE]
        # Feature: Mean Absolute Value (MAV)
        mav = np.mean(np.abs(window), axis=0)


        label = labels[i + WINDOW_SIZE // 2]

        if label != 0:
            features.append(mav)
            targets.append(label)

    return np.array(features), np.array(targets)


print("Status: Extracting features from windows...")
X, y = extract_features(X_raw, y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Status: Training Random Forest on {len(X_train)} samples...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n" + "=" * 30)
print(f"OVERALL ACCURACY: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=" * 30)
print("\nCLASSIFICATION SUMMARY (Precision/Recall per Gesture):")
print(classification_report(y_test, y_pred))

def plot_emg_signals(data, duration_secs=5, srate=2000):
    num_samples = duration_secs * srate
    time_axis = np.linspace(0, duration_secs, num_samples)
    data_slice = data[:num_samples, :]

    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Raw 12-Channel EMG Signal (First {duration_secs}s)', fontsize=14)
    colors = plt.cm.plasma(np.linspace(0, 1, 12))

    for i in range(12):
        axes[i].plot(time_axis, data_slice[:, i], color=colors[i], linewidth=0.5)
        axes[i].set_ylabel(f'Ch {i + 1}', rotation=0, labelpad=20)
        axes[i].set_yticks([])
        axes[i].grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


print("Status: Generating Signal Plot...")
plot_emg_signals(X_raw)