import cv2
import numpy as np
import time
import math

def tracking(frame, bbox, tracker, initialized):
    """
    Handles initialization and updating of the OpenCV tracker.
    """
    x0, y0, x1, y1 = bbox
    ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
    
    if not initialized:
        # Prevent zero-size bounding boxes
        tracker.init(frame, (ix, iy, max(1, iw), max(1, ih)))
        return bbox, True

    ok, new_box = tracker.update(frame)
    if ok:
        tx, ty, tw, th = [int(v) for v in new_box]
        return [tx, ty, tx + tw, ty + th], True
    
    return bbox, False

def create_tracker():
    """
    Factory to find available tracker classes across different OpenCV versions.
    """
    for attr in ['TrackerCSRT_create', 'TrackerCSRT']:
        if hasattr(cv2, attr):
            item = getattr(cv2, attr)
            return item.create() if hasattr(item, 'create') else item()
    if hasattr(cv2, 'legacy'):
        if hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    return cv2.TrackerKCF_create()

def start_soccer_tracker():
    cap = cv2.VideoCapture(0)
    tracker = create_tracker()
    
    initialized = False
    selection = []
    cropping = False
    trail = []
    
    # Physics/Timing variables
    last_pos = None
    last_time = time.time()
    velocity = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, cropping
        if not initialized:
            if event == cv2.EVENT_LBUTTONDOWN:
                selection = [x, y, x, y]
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                selection[2], selection[3] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selection[2], selection[3] = x, y
                cropping = False

    cv2.namedWindow("Soccer Tracker")
    cv2.setMouseCallback("Soccer Tracker", mouse_callback)

    print("Controls: Drag box over ball, then 'Enter'. Auto-resets if lost.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()

        if initialized:
            selection, ok = tracking(frame, selection, tracker, initialized)
            
            if ok:
                x0, y0, x1, y1 = selection
                center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
                trail.append(center)
                
                # Velocity Calculation
                curr_time = time.time()
                dt = curr_time - last_time
                if last_pos and dt > 0:
                    dist = math.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                    velocity = dist / dt
                last_pos, last_time = center, curr_time

                # Draw UI
                cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(display_frame, f"Speed: {int(velocity)} px/s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Safe Trail Drawing
                points = trail[-30:]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(display_frame, points[i-1], points[i], (0, 255, 255), 2)
            else:
                # AUTO-RESET LOGIC: Switch to re-selection mode immediately
                print("Tracking lost. Resetting to manual selection...")
                initialized = False
                selection = []
                trail = []
                last_pos = None
                tracker = create_tracker() 
        else:
            # Manual Selection Visuals
            if len(selection) == 4:
                cv2.rectangle(display_frame, (selection[0], selection[1]), 
                              (selection[2], selection[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, "Drag box and press ENTER", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Soccer Tracker", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(selection) == 4 and not initialized:
            _, initialized = tracking(frame, selection, tracker, initialized)
        elif key == ord('r'):
            initialized, selection, trail, last_pos = False, [], [], None
            tracker = create_tracker()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_soccer_tracker()
