import os
import cv2
import numpy as np
import mediapipe as mp

# ---------------- Mediapipe Setup ----------------
mp_holistic = mp.solutions.holistic

def extract_keypoints(image):
    """Extract 225 features (pose 99 + left hand 63 + right hand 63)."""
    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        pose = np.zeros(33*3)
        lh   = np.zeros(21*3)
        rh   = np.zeros(21*3)

        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z]
                             for lm in results.pose_landmarks.landmark]).flatten()

        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.left_hand_landmarks.landmark]).flatten()

        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.right_hand_landmarks.landmark]).flatten()

        return np.concatenate([pose, lh, rh])  # shape = (225,)


# ---------------- Paths ----------------
FRAMES_DIR = "processed"                # your frame folders
OUTPUT_DIR = "landmark_sequences"    # where .npy files will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- Main Processing ----------------
def process_all_sequences():
    for class_name in os.listdir(FRAMES_DIR):
        class_path = os.path.join(FRAMES_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for seq_folder in os.listdir(class_path):
            seq_path = os.path.join(class_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            frames = sorted(os.listdir(seq_path))[:20]  # only 20 frames

            if len(frames) < 20:
                print(f"❌ Skipped {seq_folder}: less than 20 frames")
                continue

            sequence = []

            for frame_name in frames:
                frame_path = os.path.join(seq_path, frame_name)
                image = cv2.imread(frame_path)

                keypoints = extract_keypoints(image)
                sequence.append(keypoints)

            sequence = np.array(sequence)  # shape = (20, 225)

            np.save(os.path.join(output_class_dir, seq_folder), sequence)

            print(f"✅ Saved {seq_folder}: shape = {sequence.shape}")

    print("\n🎉 All sequences converted successfully!")


if __name__ == "__main__":
    process_all_sequences()
