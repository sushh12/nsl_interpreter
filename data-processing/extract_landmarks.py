import cv2
import mediapipe as mp
import os
import numpy as np

dataset_dir = "processed_dataset"
output_dir = "landmark_sequences"
max_frames = 30

mp_hands = mp.solutions.hands
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
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            frame_files = sorted(os.listdir(video_path))
            for f in frame_files[:max_frames]:
                img_path = os.path.join(video_path, f)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0]
                    coords = []
                    for p in lm.landmark:
                        coords += [p.x, p.y, p.z]
                    sequence.append(coords)
                else:
                    sequence.append([0]*63)

        sequence = np.array(sequence)
        np.save(os.path.join(output_class_path, video_folder + ".npy"), sequence)
        print(f"Saved landmark sequence for {video_folder}")
print("✅ Landmark extraction completed")
