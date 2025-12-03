import numpy as np
import pandas as pd

df = pd.read_csv("labels.csv")

TARGET_LEN = 20

for path in df["file_path"]:
    p = path.replace("\\", "/")
    arr = np.load(p)

    if arr.shape[0] < TARGET_LEN:
        # Pad with last frame
        last = arr[-1:]
        pad_count = TARGET_LEN - arr.shape[0]
        padded = np.vstack([arr, np.repeat(last, pad_count, axis=0)])

        np.save(p, padded)
        print(f"🔧 Fixed (padded): {p} from {arr.shape} → {padded.shape}")

    elif arr.shape[0] > TARGET_LEN:
        # Trim extra frames
        trimmed = arr[:TARGET_LEN]
        np.save(p, trimmed)
        print(f"✂ Trimmed: {p} from {arr.shape} → {trimmed.shape}")
