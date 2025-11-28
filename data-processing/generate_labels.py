import os
import csv

dataset_dir = "landmark_sequences"
output_csv = "labels.csv"

rows = []

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # loop through files
    for file_name in os.listdir(class_path):
        if file_name.endswith(".npy"):          # only select landmark files
            full_path = os.path.join(class_path, file_name)
            rows.append([full_path.replace("\\", "/"), class_name])

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "label"])
    writer.writerows(rows)

print(f"✅ Labels saved in {output_csv}. Total sequences: {len(rows)}")
