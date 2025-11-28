import cv2
import mediapipe as mp
import os
import numpy as np

dataset_dir = "processed_Consonant"
output_dir = "landmark_sequences"
max_frames = 20

mp_holistic = mp.solutions.holistic
os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for video_folder in os.listdir(class_path):
        video_path = os.path.join(class_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        sequence = []

        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            frame_files = sorted(os.listdir(video_path))

            for f in frame_files[:max_frames]:
                img_path = os.path.join(video_path, f)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = holistic.process(img_rgb)

                # ---- Extract Hands Only ----
                left_hand = []
                right_hand = []

                # Left hand (21 landmarks)
                if result.left_hand_landmarks:
                    for lm in result.left_hand_landmarks.landmark:
                        left_hand += [lm.x, lm.y, lm.z]
                else:
                    left_hand = [0] * 63   # 21*3

                # Right hand (21 landmarks)
                if result.right_hand_landmarks:
                    for lm in result.right_hand_landmarks.landmark:
                        right_hand += [lm.x, lm.y, lm.z]
                else:
                    right_hand = [0] * 63   # 21*3

                # Combine both hands → 126 values
                frame_landmarks = left_hand + right_hand
                sequence.append(frame_landmarks)

        sequence = np.array(sequence)
        np.save(os.path.join(output_class_path, video_folder + ".npy"), sequence)

        print(f"Saved landmark sequence for {video_folder}")

print("✅ Landmark extraction completed using Mediapipe Holistic (hands only)")
