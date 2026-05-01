import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

file_names = ['S1_E1_A1.mat', 'S1_E2_A1.mat', 'S1_E3_A1.mat']
all_data = []

for file in file_names:
    mat = scipy.io.loadmat(file)
    emg = mat['emg']

    labels = mat.get('stimulus', mat.get('restimulus', np.zeros((emg.shape[0], 1))))
    columns = [f'Ch_{i + 1}' for i in range(12)]
    df_temp = pd.DataFrame(emg, columns=columns)
    df_temp['gesture'] = labels.flatten()
    all_data.append(df_temp)

df = pd.concat(all_data, ignore_index=True)
print("Dataframe Head:")
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_ninapro_file(file_path):
    mat = scipy.io.loadmat(file_path)
    emg = mat['emg']

    if 'stimulus' in mat:
        labels = mat['stimulus']
    elif 'restimulus' in mat:
        labels = mat['restimulus']
    else:

        labels = None
        for key in mat.keys():
            if not key.startswith('__') and mat[key].shape[0] == emg.shape[0]:
                labels = mat[key]
                break
        if labels is None:
            raise KeyError(f"Could not find a label variable in {file_path}")

    return emg, labels.flatten()


files = ['S1_E1_A1.mat', 'S1_E2_A1.mat', 'S1_E3_A1.mat']
all_emg = []
all_labels = []

for f in files:
    try:
        e, l = load_ninapro_file(f)
        all_emg.append(e)
        all_labels.append(l)
        print(f"Loaded {f}: {e.shape}")
    except Exception as err:
        print(f"Error loading {f}: {err}")


X_raw = np.vstack(all_emg)
y_raw = np.concatenate(all_labels)


X_rectified = np.abs(X_raw)


def create_windows(data, labels, window_size=200, step=100):
    features = []
    targets = []

    for i in range(0, len(data) - window_size, step):
        window = data[i: i + window_size]

        mav = np.mean(window, axis=0)

        label = labels[i + window_size // 2]


        if label != 0:
            features.append(mav)
            targets.append(label)

    return np.array(features), np.array(targets)


print("Processing windows...")
X_features, y_labels = create_windows(X_rectified, y_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)
print(f"Training on {len(X_train)} samples...")
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\nSuccess! Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

mat_data = scipy.io.loadmat('S1_E1_A1.mat')
emg_signal = mat_data['emg']
sampling_rate = 2000
seconds_to_plot = 5
num_samples = seconds_to_plot * sampling_rate
data_slice = emg_signal[:num_samples, :]
time_axis = np.linspace(0, seconds_to_plot, num_samples)
fig, axes = plt.subplots(12, 1, figsize=(12, 15), sharex=True)
fig.suptitle(f'12-Channel EMG Signal (First {seconds_to_plot}s)', fontsize=16)

colors = plt.cm.viridis(np.linspace(0, 1, 12))

for i in range(12):
    axes[i].plot(time_axis, data_slice[:, i], color=colors[i], linewidth=0.7)
    axes[i].set_ylabel(f'Ch {i+1}', rotation=0, labelpad=20)
    axes[i].set_yticks([]) # Keeps it clean

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()