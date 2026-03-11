import cv2
import numpy as np
import torch
import os
import urllib.request
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_CHECKPOINT = "sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"  # This must match your installed sam2/configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Color Configuration (BGR)
BG_WHITE = [255, 255, 255]
BG_BLACK = [0, 0, 0]
CURRENT_BG = BG_WHITE # <-- CHANGE COLOR HERE

def download_weights():
    """Downloads the SAM 2 weights if they don't exist."""
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"Downloading {SAM2_CHECKPOINT}... this may take a minute.")
        urllib.request.urlretrieve(CHECKPOINT_URL, SAM2_CHECKPOINT)
        print("Download complete.")

def run_segmentation(img_path):
    """Applies SAM 2 to the cropped image."""
    # 1. Load Model
    download_weights()
    model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(model)

    # 2. Process Image
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # 3. Prompting (Center Point)
    h, w = image.shape[:2]
    input_points = np.array([[w // 2, h // 2]])
    input_labels = np.array([1]) 

    # 4. Generate Mask
    print("Segmenting face...")
    masks, _, _ = predictor.predict(input_points, input_labels, multimask_output=False)
    mask = masks[0].astype(bool)

    # 5. Apply Background Color (Step-by-Step)
    # Create canvas
    output = np.full(image.shape, CURRENT_BG, dtype=np.uint8)
    # Transfer face pixels
    output[mask] = image[mask]

    # 6. Save and Show
    cv2.imwrite("segmented_face.png", output)
    cv2.imshow("Result (Press any key to close)", output)
    cv2.waitKey(0)

def start_camera_workflow():
    cap = cv2.VideoCapture(0)
    frozen = False
    frozen_frame = None
    selection = []
    cropping = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, cropping
        if frozen:
            if event == cv2.EVENT_LBUTTONDOWN:
                selection = [(x, y)]
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                if len(selection) == 2: selection[1] = (x, y)
                else: selection.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                selection[1] = (x, y)
                cropping = False

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)
    print("S: Freeze | Enter: Segment | Q: Quit")

    while True:
        ret, frame = cap.read() if not frozen else (True, frozen_frame)
        if not ret: break
        
        display = frame.copy()
        if len(selection) == 2:
            cv2.rectangle(display, selection[0], selection[1], (255, 0, 0), 2)

        cv2.imshow("Camera", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            frozen = not frozen
            frozen_frame = frame.copy() if frozen else None
            selection = []
        elif key == 13 and frozen and len(selection) == 2:
            y_coords = sorted([selection[0][1], selection[1][1]])
            x_coords = sorted([selection[0][0], selection[1][0]])
            crop = frozen_frame[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
            cv2.imwrite("cropped.png", crop)
            cap.release()
            cv2.destroyAllWindows()
            run_segmentation("cropped.png")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_workflow()
