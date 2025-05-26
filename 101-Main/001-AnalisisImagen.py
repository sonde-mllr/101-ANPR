import cv2
import pytesseract
import numpy as np

# Path to the image file
IMAGE_PATH = '201-Imagenes/matricula.jpg'  # Change this to your image path

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Image not found.")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal with bilateral filter
filtered = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(filtered, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate_contour = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(approx)
        # Typical European plate: aspect ratio between 2 and 6, area > 2000
        if 2 < aspect_ratio < 6 and area > 2000:
            plate_contour = approx
            break

if plate_contour is None:
    print("Number plate contour not found.")
    exit(1)

# Mask everything except the plate
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [plate_contour], 0, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)

# Crop the plate area
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

# OCR with pytesseract
custom_config = r'--oem 3 --psm 11'
language = r'spa'
text = pytesseract.image_to_string(cropped, config=custom_config,lang=language)
print("Detected Number Plate Text:", text.strip())

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Masked Plate", masked)
cv2.imshow("Cropped Plate", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
