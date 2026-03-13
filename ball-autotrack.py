import cv2
import numpy as np
import time
import math

def tracking(frame, bbox, tracker, initialized):
    x0, y0, x1, y1 = bbox
    ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
    
    if not initialized:
        tracker.init(frame, (ix, iy, max(1, iw), max(1, ih)))
        return bbox, True

    ok, new_box = tracker.update(frame)
    if ok:
        tx, ty, tw, th = [int(v) for v in new_box]
        return [tx, ty, tx + tw, ty + th], True
    
    return bbox, False

def create_tracker():
    for attr in ['TrackerCSRT_create', 'TrackerCSRT']:
        if hasattr(cv2, attr):
            item = getattr(cv2, attr)
            return item.create() if hasattr(item, 'create') else item()
    return cv2.TrackerKCF_create()

def attempt_reacquisition(frame, template):
    """
    Uses template matching to find the saved crop in the current frame.
    Returns (new_bbox, success)
    """
    if template is None:
        return None, False
    
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Threshold for a "good" match (0.7 is a solid starting point)
    if max_val > 0.7:
        h, w = template.shape[:2]
        tx, ty = max_loc
        return [tx, ty, tx + w, ty + h], True
    return None, False

def start_soccer_tracker():
    cap = cv2.VideoCapture(0)
    tracker = create_tracker()
    
    initialized = False
    selection = []
    cropping = False
    trail = []
    
    last_pos = None
    last_time = time.time()
    velocity = 0
    
    trigger_init = False
    saved_crop = None
    crop_visible = False
    reacquiring = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, cropping, trigger_init
        if not initialized:
            if event == cv2.EVENT_LBUTTONDOWN:
                selection = [x, y, x, y]
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                selection[2], selection[3] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selection[2], selection[3] = x, y
                cropping = False
                if abs(selection[2] - selection[0]) > 5 and abs(selection[3] - selection[1]) > 5:
                    trigger_init = True

    cv2.namedWindow("Soccer Tracker")
    cv2.setMouseCallback("Soccer Tracker", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()

        # Handle Initial Manual Lock
        if trigger_init and not initialized and len(selection) == 4:
            x0, y0, x1, y1 = selection
            ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
            if iw > 0 and ih > 0:
                saved_crop = frame[iy:iy+ih, ix:ix+iw].copy()
                _, initialized = tracking(frame, selection, tracker, initialized)
            trigger_init = False

        if initialized:
            selection, ok = tracking(frame, selection, tracker, initialized)
            
            if ok:
                reacquiring = False
                x0, y0, x1, y1 = selection
                center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
                trail.append(center)
                
                curr_time = time.time()
                dt = curr_time - last_time
                if last_pos and dt > 0:
                    dist = math.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                    velocity = dist / dt
                last_pos, last_time = center, curr_time

                cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(display_frame, f"Speed: {int(velocity)} px/s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                points = trail[-30:]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(display_frame, points[i-1], points[i], (0, 255, 255), 2)
            else:
                # Lock lost, attempt reacquisition
                reacquiring = True
                new_selection, found = attempt_reacquisition(frame, saved_crop)
                if found:
                    selection = new_selection
                    tracker = create_tracker()
                    _, initialized = tracking(frame, selection, tracker, False)
                    reacquiring = False
                else:
                    cv2.putText(display_frame, "LOCK LOST - Searching...", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            if len(selection) == 4:
                cv2.rectangle(display_frame, (selection[0], selection[1]), 
                              (selection[2], selection[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, "Drag to start", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Soccer Tracker", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(selection) == 4 and not initialized:
            trigger_init = True
        elif key == ord('s'):
            if saved_crop is not None:
                if not crop_visible:
                    cv2.imshow("Saved Crop", saved_crop)
                    crop_visible = True
                else:
                    try: cv2.destroyWindow("Saved Crop")
                    except: pass
                    crop_visible = False
        elif key == ord('r'):
            initialized, selection, trail, last_pos, trigger_init, reacquiring = False, [], [], None, False, False
            tracker = create_tracker()
            try: cv2.destroyWindow("Saved Crop")
            except: pass
            crop_visible = False
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_soccer_tracker()
