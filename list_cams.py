"""Show a 2-second preview of each camera index so you can find the iPhone (Continuity Camera).
The index can change between reboots. Needs camera permission for the app running Python."""
import cv2
found = False
for i in range(4):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    if not ok:
        cap.release(); continue
    found = True
    print(f"index {i}: {frame.shape[1]}x{frame.shape[0]} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    cv2.imshow(f"camera {i}  (any key = next)", frame); cv2.waitKey(2000); cv2.destroyAllWindows(); cap.release()
if not found:
    print("no camera could be opened: allow Camera access for this app in System Settings > Privacy & Security")
