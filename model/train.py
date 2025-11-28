import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import joblib

# ---------------- SETTINGS ----------------
data_dir = "landmark_sequences"
max_frames = 20     # Your model expects 30 frames
feature_size = 126   # 21 hand landmarks * (x, y, z) Pose: 33 × 3 = 99
                                                    # Right Hand: 21 × 3 = 63
                                                    # Left Hand: 21 × 3 = 63

# ---------------- FIX SEQUENCE LENGTH ----------------
def fix_sequence_length(seq, target_len=30):
    """
    Pads or trims sequence to exactly target_len frames.
    """
    if len(seq) < target_len:
        pad_len = target_len - len(seq)
        padding = np.zeros((pad_len, seq.shape[1]))
        seq = np.vstack((seq, padding))
    return seq[:target_len]

# ---------------- LOAD DATA ----------------
X, y = [], []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(class_path, file))

            # ⭐ FIX THE SHAPE HERE
            seq = fix_sequence_length(seq, max_frames)

            X.append(seq)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

print("Loaded dataset shape:", X.shape)   # Should be (N, 30, 63)

# ---------------- LABEL ENCODING ----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, "label_encoder.pkl")

# ---------------- TRAIN/TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, shuffle=True
)

num_classes = len(np.unique(y_train))

# ---------------- DATA AUGMENTATION ----------------
def augment_sequence(seq):
    seq_aug = seq.copy()

    # small random noise
    noise = np.random.normal(0, 0.002, seq_aug.shape)
    seq_aug += noise

    # scale pose slightly
    scale = np.random.uniform(0.98, 1.02)
    seq_aug[:, :feature_size] *= scale

    # shift landmarks slightly
    shift = np.random.uniform(-0.01, 0.01)
    seq_aug[:, :feature_size] += shift

    return seq_aug

# Generate augmented data
augmented_X, augmented_y = [], []

for seq, label in zip(X_train, y_train):
    # original
    augmented_X.append(seq)
    augmented_y.append(label)

    # 2 augmented copies
    for _ in range(2):
        augmented_X.append(augment_sequence(seq))
        augmented_y.append(label)

X_train_aug = np.array(augmented_X)
y_train_aug = np.array(augmented_y)

print("Augmented training shape:", X_train_aug.shape)

# ---------------- MODEL ----------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(max_frames, feature_size)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CLASS WEIGHTS ----------------
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_aug),
    y=y_train_aug
)

class_weights = dict(enumerate(weights))

# ---------------- EARLY STOPPING ----------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ---------------- TRAIN ----------------
history = model.fit(
    X_train_aug, y_train_aug,
    batch_size=16,
    epochs=50,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ---------------- SAVE MODEL ----------------
model.save("model.keras")

print("\n✅ Model training completed successfully with padding, augmentation and class balancing.")
