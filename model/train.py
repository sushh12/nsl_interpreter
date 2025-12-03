import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

data_dir = "landmark_sequences"

def load_dataset():
    X, y = [], []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(class_path, file))

                if arr.shape == (20, 225):
                    X.append(arr)
                    y.append(class_name)
                else:
                    print("Skipping invalid file:", file, arr.shape)

    X = np.array(X)
    y = np.array(y)

    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    return X, y


def build_model(num_classes):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(20, 225)),
        Dropout(0.4),
        LSTM(256),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    X, y = load_dataset()

    # Encode labels
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    model = build_model(y.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16
    )

    model.save("sign_model.h5")
    print("✅ Model saved as sign_model.h5")


if __name__ == "__main__":
    main()
