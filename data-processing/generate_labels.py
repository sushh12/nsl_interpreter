import os
import csv

dataset_dir = "processed_consonant"
output_csv = "labels.csv"

rows = []

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for video_folder in os.listdir(class_path):
        video_path = os.path.join(class_path, video_folder)
        rows.append([video_path, class_name])

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "label"])
    writer.writerows(rows)

print(f"✅ Labels saved in {output_csv}. Total sequences: {len(rows)}")
