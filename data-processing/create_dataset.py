import os
import cv2
import numpy as np
from extract_landmarks import extract_keypoints

ROOT = "processed"     # your dataset directory
SEQUENCE_LENGTH = 20   # 20 frames per video

labels = sorted(os.listdir(ROOT))
label_map = {label:i for i, label in enumerate(labels)}

X = []
y = []

for label in labels:
    class_path = os.path.join(ROOT, label)

    for video_folder in os.listdir(class_path):
        frames_path = os.path.join(class_path, video_folder)
        if not os.path.isdir(frames_path):
            continue

        sequence = []
        frames = sorted(os.listdir(frames_path))

        if len(frames) != SEQUENCE_LENGTH:
            print(f"Skipping {frames_path} because frame count != 20")
            continue

        for frame_name in frames:
            frame_path = os.path.join(frames_path, frame_name)
            img = cv2.imread(frame_path)

            keypoints = extract_keypoints(img)
            sequence.append(keypoints)

        X.append(sequence)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save("X_keypoints.npy", X)
np.save("y_labels.npy", y)

print("Dataset created!")
print("X shape:", X.shape)   # (samples, 20, 225)
print("y shape:", y.shape)
