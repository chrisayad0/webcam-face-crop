import cv2

def start_cropper():
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
                if len(selection) == 2:
                    selection[1] = (x, y)
                else:
                    selection.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                selection[1] = (x, y)
                cropping = False

    cv2.namedWindow("Cropper")
    cv2.setMouseCallback("Cropper", mouse_callback)

    print("Controls:\n's' - Freeze/Unfreeze\n'Enter' - Save Crop\n'q' - Quit")

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret: break
            display_frame = frame.copy()
            cv2.putText(display_frame, "Live: Press 's' to freeze", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frame = frozen_frame.copy()
            cv2.putText(display_frame, "Frozen: Click/Drag to crop, then 'Enter'", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw highlight
        if len(selection) == 2:
            cv2.rectangle(display_frame, selection[0], selection[1], (255, 0, 0), 2)

        cv2.imshow("Cropper", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
            else:
                selection = []

        elif key == 13 and frozen and len(selection) == 2: # Enter Key
            x1, y1 = selection[0]
            x2, y2 = selection[1]
            # Ensure proper slicing regardless of drag direction
            iy1, iy2 = sorted([y1, y2])
            ix1, ix2 = sorted([x1, x2])
            
            cropped_img = frozen_frame[iy1:iy2, ix1:ix2]
            
            if cropped_img.size > 0:
                cv2.imwrite("cropped.png", cropped_img)
                print("Image saved as cropped.png")
                cv2.imshow("Final Crop", cropped_img)
                cv2.waitKey(0) # Wait for key before closing
                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_cropper()
