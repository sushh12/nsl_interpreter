# real-time.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import tensorflow as tf
import joblib
import time
import os

# -----------------------------
# Settings (matches your train)
# -----------------------------
MODEL_PATH = "model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_FRAMES = 20
FEATURE_SIZE = 126   # left(63) + right(63)
SMOOTH_WINDOW = 7    # majority vote window
CONF_THRESHOLD = 0.60  # show label only when confident

# Optional debug flags
DEBUG_PRINT_PRED = False   # prints raw probs (top classes)
SHOW_FPS = True

# -----------------------------
# Load model and label encoder
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

print("Loaded model:", MODEL_PATH)
print("Label classes:", len(le.classes_), "classes")

# -----------------------------
# Mediapipe Holistic (same as extraction)
# -----------------------------
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=True,          # IMPORTANT: match extraction pipeline
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Utility functions
# -----------------------------
def extract_2hand_features(results):
    """
    Return a list of 126 floats: left_hand(63) + right_hand(63)
    Order matches your extraction script.
    """
    left_hand = []
    right_hand = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            left_hand.extend([lm.x, lm.y, lm.z])
    else:
        left_hand = [0.0] * 63

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            right_hand.extend([lm.x, lm.y, lm.z])
    else:
        right_hand = [0.0] * 63

    return left_hand + right_hand  # length = 126

def pad_sequence_to_length(seq_list, target_len=MAX_FRAMES):
    """
    If seq shorter than target_len, pad by repeating last frame.
    If empty, returns zeros.
    """
    if len(seq_list) == 0:
        return np.zeros((target_len, FEATURE_SIZE), dtype=np.float32)
    arr = np.array(seq_list, dtype=np.float32)
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    # pad by repeating last frame
    last = arr[-1]
    pad_count = target_len - arr.shape[0]
    pads = np.tile(last, (pad_count, 1))
    return np.vstack([arr, pads])

def get_hand_square_bbox(hand_landmarks, frame_shape, pad=10):
    """
    Returns square bbox coords (x1,y1,x2,y2) in pixel space for a single hand.
    If hand_landmarks is None, returns None.
    """
    if not hand_landmarks:
        return None
    h, w, _ = frame_shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x_min = int(min(xs) * w)
    x_max = int(max(xs) * w)
    y_min = int(min(ys) * h)
    y_max = int(max(ys) * h)

    box_w = x_max - x_min
    box_h = y_max - y_min
    size = max(box_w, box_h) + pad * 2
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return x1, y1, x2, y2

# -----------------------------
# Main loop
# -----------------------------
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=MAX_FRAMES)
pred_buffer = deque(maxlen=SMOOTH_WINDOW)

prev_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(img_rgb)

        # draw detected landmarks for visual feedback
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # draw square boxes around each hand (if present)
        left_box = get_hand_square_bbox(results.left_hand_landmarks, frame.shape)
        right_box = get_hand_square_bbox(results.right_hand_landmarks, frame.shape)
        if left_box:
            cv2.rectangle(frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (255, 0, 0), 2)
        if right_box:
            cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 0, 255), 2)

        # get features for this frame
        feat = extract_2hand_features(results)
        sequence.append(feat)

        predicted_label = " "
        confidence = 0.0

        # Predict when we have enough frames (pad if slightly shorter)
        if len(sequence) == MAX_FRAMES:
            X = np.array(sequence, dtype=np.float32).reshape(1, MAX_FRAMES, FEATURE_SIZE)

            # model.predict with verbose=0 avoids console spam
            pred = model.predict(X, verbose=0)[0]   # shape (num_classes,)
            class_id = int(np.argmax(pred))
            confidence = float(pred[class_id])

            # smoothing: majority vote in buffer
            pred_buffer.append(class_id)
            if len(pred_buffer) == SMOOTH_WINDOW:
                most_common = Counter(pred_buffer).most_common(1)[0][0]
                # choose label only if confidence above threshold OR majority is stable
                if confidence >= CONF_THRESHOLD:
                    final_id = most_common
                else:
                    final_id = most_common  # you can require both if you want
            else:
                final_id = class_id

            # convert to label string using label encoder
            predicted_label = le.inverse_transform([final_id])[0]

            if DEBUG_PRINT_PRED:
                # Print top 5 probabilities
                top5_idx = np.argsort(pred)[-5:][::-1]
                print("Top5:", [(le.inverse_transform([i])[0], float(pred[i])) for i in top5_idx])

        # show prediction and confidence
        disp_text = predicted_label if predicted_label else "..."
        cv2.putText(frame, f"{disp_text} ({confidence:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # FPS
        if SHOW_FPS:
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("NSL Interpreter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # debug: press 's' to save last sequence for inspection
        if key == ord('s'):
            arr = np.array(sequence)
            path = os.path.join("debug_saved_seqs", f"seq_{int(time.time())}.npy")
            os.makedirs("debug_saved_seqs", exist_ok=True)
            np.save(path, arr)
            print("Saved sequence:", path)

finally:
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
