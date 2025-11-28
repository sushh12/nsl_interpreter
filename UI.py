import cv2
import numpy as np

# ------------------------------------
# UI DIMENSIONS
# ------------------------------------

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

CAM_WIDTH = 640
CAM_HEIGHT = 360

DIALOG_WIDTH = 500
DIALOG_HEIGHT = 70

# ------------------------------------
# Create gradient background
# ------------------------------------
def create_gradient_background(width, height, color1, color2):
    """Generates a smooth horizontal gradient."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        alpha = x / width
        gradient[:, x] = (1 - alpha) * np.array(color1) + alpha * np.array(color2)
    return gradient


# Soft gradient colors (BGR)
COLOR_LEFT = (250, 240, 255)  
COLOR_RIGHT = (220, 20, 255)

BACKGROUND = create_gradient_background(WINDOW_WIDTH, WINDOW_HEIGHT, COLOR_LEFT, COLOR_RIGHT)


# ------------------------------------
# Start Camera
# ------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

dialog_text = "Text"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame
    # Resize camera frame
    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

    # Copy gradient background each frame
    canvas = BACKGROUND.copy()

    # ------------------------------------
    # Draw Heading above the camera
    # ------------------------------------
    heading_text = "NEPALI SIGN LANGUAGE"

    cv2.putText(
        canvas,
        heading_text,
        (120, 80),  # x, y position
        cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC,  # fancy cursive-like font
        1.6,  # font scale
        (0, 0, 0),  # white color
        2,  # thickness
        cv2.LINE_AA
    )

    # ------------------------------------
    # Place Camera Frame (centered)
    # ------------------------------------
    cam_x = (WINDOW_WIDTH - CAM_WIDTH) // 2
    cam_y = 120

    canvas[cam_y:cam_y + CAM_HEIGHT, cam_x:cam_x + CAM_WIDTH] = frame

  
    # ------------------------------------
    # Create Dialog Box
    # ------------------------------------
    dialog_box = np.zeros((DIALOG_HEIGHT, DIALOG_WIDTH, 3), dtype=np.uint8)
    dialog_box[:] = (255, 255, 255)  # white box

    # Put text inside dialog
    cv2.putText(
        dialog_box,
        dialog_text,
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Center dialog box under camera
    dialog_x = (WINDOW_WIDTH - DIALOG_WIDTH) // 2
    dialog_y = cam_y + CAM_HEIGHT + 40

    canvas[dialog_y:dialog_y + DIALOG_HEIGHT, dialog_x:dialog_x + DIALOG_WIDTH] = dialog_box

    # ------------------------------------
    # Show Final UI
    # ------------------------------------
    cv2.imshow("Nepali Sign Language UI", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
