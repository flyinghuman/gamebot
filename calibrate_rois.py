# calibrate_rois.py
import cv2, numpy as np, time, sys, subprocess

SERIAL = "emulator-5554"  # adjust as needed
ADB = ["adb", "-s", SERIAL]

def adb_screenshot():
    res = subprocess.run(ADB + ["exec-out", "screencap", "-p"], capture_output=True, timeout=10)
    img = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
    return img

drawing = False
pt1 = None
rect = None
img = None
disp = None

def on_mouse(event, x, y, flags, param):
    global drawing, pt1, rect, disp
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1 = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        disp = img.copy()
        cv2.rectangle(disp, pt1, (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = pt1
        x2, y2 = x, y
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        H, W = img.shape[:2]
        rx1, ry1, rx2, ry2 = x1/W, y1/H, x2/W, y2/H
        print(f"Normalized ROI: [{rx1:.4f}, {ry1:.4f}, {rx2:.4f}, {ry2:.4f}]")
        disp = img.copy()
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

def main():
    global img, disp
    img = adb_screenshot()
    disp = img.copy()
    cv2.namedWindow("ROI Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Calibration", on_mouse)
    print('Instructions: drag with the mouse to mark an ROI. "r" = reload, "q" = quit.')
    while True:
        cv2.imshow("ROI Calibration", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            img = adb_screenshot()
            disp = img.copy()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()