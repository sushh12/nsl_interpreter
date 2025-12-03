# extracting frames from raw video dataset
import os
import cv2
import math

# -------- SETTINGS --------
raw_dataset = "NSL_Consonant"         # Your consonant dataset folder
output_dir = "processed"    # Output folder for consonants
frame_count = 20                      
frame_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

VIDEO_EXTS = [".mov", ".mp4", ".avi", ".mkv"]

video_id_counter = {}

for root, _, files in os.walk(raw_dataset):
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
            output_dir, 
            label, 
            f"{parts[0]}{label}{video_id_counter[label]:02d}"
        )
        os.makedirs(video_folder, exist_ok=True)

        video_path = os.path.join(root, file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_frame_count))

        if total_frames < frame_count:
            frame_indices = list(range(total_frames))
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
                frame_resized = cv2.resize(frame, frame_size)
                frame_filename = os.path.join(video_folder, f"{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame_resized)
                saved_count += 1

            frame_idx += 1

        cap.release()
        print(f"Processed {file} -> {saved_count} frames in {video_folder}")

print("✅ All consonant videos processed successfully!")