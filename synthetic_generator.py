import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

def generate_synthetic_question(q_num, math_text, options, out_path="synthetic_q.jpg"):
    # 1. Base monitor screen
    width, height = 1920, 1080
    img = Image.new('RGB', (width, height), color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    
    # 2. Draw mock header
    draw.rectangle([0, 0, width, 80], fill=(40, 60, 120))
    
    try:
        font_large = ImageFont.truetype("Arial", 40)
        font_medium = ImageFont.truetype("Arial", 30)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        
    draw.text((50, 20), "Mathematics Examination - Section A", fill=(255, 255, 255), font=font_large)
    
    # 3. Draw Question Stem
    stem = f"Q{q_num}. {math_text}"
    draw.text((100, 200), stem, fill=(0, 0, 0), font=font_large)
    
    # 4. Draw Options
    y_offset = 400
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        draw.text((120, y_offset), f"{letter}.   {opt}", fill=(0, 0, 0), font=font_medium)
        y_offset += 100
        
    # 5. Convert to OpenCV to apply physical distortions (camera simulation)
    cv_img = np.array(img)
    cv_img = cv_img[:, :, ::-1].copy() 
    
    # Add perspective warp
    pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
    
    # Randomly distort corners by 50-100 pixels
    d = 80
    pts2 = np.float32([
        [random.randint(0,d), random.randint(0,d)],
        [width-random.randint(0,d), random.randint(0,d)],
        [random.randint(0,d), height-random.randint(0,d)],
        [width-random.randint(0,d), height-random.randint(0,d)]
    ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(cv_img, M, (width, height), borderValue=(50, 50, 50))
    
    # Add glare
    glare_overlay = warped.copy()
    cv2.circle(glare_overlay, (width//2, height//3), 400, (255, 255, 255), -1)
    warped = cv2.addWeighted(warped, 0.8, glare_overlay, 0.2, 0)
    
    # Add noise
    noise = np.random.normal(0, 15, warped.shape).astype(np.uint8)
    warped = cv2.add(warped, noise)
    
    # Save
    cv2.imwrite(out_path, warped)
    print(f"Generated synthetic image: {out_path}")

if __name__ == "__main__":
    generate_synthetic_question(
        11, 
        "If det [[2, x], [3, 5]] = 4, then the value of x is:",
        ["2/3", "4/3", "4", "2"],
        "synthetic_q11.jpg"
    )
