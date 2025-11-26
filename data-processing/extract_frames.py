import os
import cv2
import math

# -------- SETTINGS --------
raw_dataset = "NSL_Vowel"             # Your original dataset folder
output_dir = "processed_dataset1"      # Where processed frames will go
frame_count = 30                       # Fixed frames per video
frame_size = (224, 224)                # Resize frames

os.makedirs(output_dir, exist_ok=True)

# Supported video extensions
video_exts = [".mov", ".mp4", ".avi", ".mkv"]

video_id_counter = {}

# Walk through the dataset
for root, _, files in os.walk(raw_dataset):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in video_exts:
            continue
        
        # Extract label from filename
        parts = file.split("_")
        if len(parts) < 2:
            continue
        label = os.path.splitext(parts[1])[0].upper()  # "S1_A.mov" -> "A"

        # Prepare output folder
        video_id_counter[label] = video_id_counter.get(label, 0) + 1
        video_folder = os.path.join(output_dir, label, f"{parts[0]}_{label}_{video_id_counter[label]:02d}")
        os.makedirs(video_folder, exist_ok=True)

        # Open video
        video_path = os.path.join(root, file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frame_count:
            frame_indices = list(range(total_frames))  # take all, will pad later
        else:
            step = total_frames / frame_count
            frame_indices = [math.floor(step * i) for i in range(frame_count)]

        frame_idx = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                # Resize frame
                frame_resized = cv2.resize(frame, frame_size)
                # Save frame
                frame_filename = os.path.join(video_folder, f"{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame_resized)
                saved_count += 1
            frame_idx += 1

        cap.release()
        print(f"Processed {file} -> {saved_count} frames in {video_folder}")

print("✅ All videos processed successfully!")
