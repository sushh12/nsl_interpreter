import os
import cv2
import math

# -------- SETTINGS --------
RAW_DATASET = "NSL_Consonant"         # Your consonant dataset folder
OUTPUT_DIR = "processed_consonant"    # Output folder for consonants
FRAME_COUNT = 20                      
FRAME_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_EXTS = [".mov", ".mp4", ".avi", ".mkv"]

video_id_counter = {}

for root, _, files in os.walk(RAW_DATASET):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in VIDEO_EXTS:
            continue
        
        parts = file.split("_")
        if len(parts) < 2:
            continue
        
        # Example: "S1_KA.mov" → "KA"
        label = os.path.splitext(parts[1])[0].upper()

        video_id_counter[label] = video_id_counter.get(label, 0) + 1
        video_folder = os.path.join(
            OUTPUT_DIR, 
            label, 
            f"{parts[0]}{label}{video_id_counter[label]:02d}"
        )
        os.makedirs(video_folder, exist_ok=True)

        video_path = os.path.join(root, file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < FRAME_COUNT:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / FRAME_COUNT
            frame_indices = [math.floor(step * i) for i in range(FRAME_COUNT)]

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_indices:
                frame_resized = cv2.resize(frame, FRAME_SIZE)
                frame_filename = os.path.join(video_folder, f"{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame_resized)
                saved_count += 1

            frame_idx += 1

        cap.release()
        print(f"Processed {file} -> {saved_count} frames in {video_folder}")

print("✅ All consonant videos processed successfully!")