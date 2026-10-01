import cv2
import numpy as np
from PIL import Image, ImageOps
import sys

def rectify_screen(image_path, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return False
        
    orig = img.copy()
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We want to find the bright monitor area against the dark background.
    # blurring to remove moire / noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_cnt = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    if screen_cnt is None:
        print("Could not find a 4-point contour for the screen.")
        # fallback to bounding box of largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        screen_cnt = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    else:
        screen_cnt = screen_cnt.reshape(4, 2)
        
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = screen_cnt.sum(axis=1)
    rect[0] = screen_cnt[np.argmin(s)]
    rect[2] = screen_cnt[np.argmax(s)]
    diff = np.diff(screen_cnt, axis=1)
    rect[1] = screen_cnt[np.argmin(diff)]
    rect[3] = screen_cnt[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    cv2.imwrite(out_path, warped)
    print(f"Saved rectified image to {out_path}")
    return True

if __name__ == "__main__":
    rectify_screen("q11_rot270.jpg", "q11_rectified.jpg")
