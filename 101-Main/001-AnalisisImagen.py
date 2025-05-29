""" Este códgo ha sido implementado a partir de otros proyectos de github y tutoriales de internet.

    Algunos de estos son:
        - https://github.com/hanishmangalasseri/ANPR-ADAS-Project/tree/master
            -> Principal para la estructura en funciones
        - https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
        - https://github.com/GuiltyNeuron/ANPR
            -> El objetivo final del proyecto es que se parezca a projecto Dutch_anpr
    

    Lo modifiqué que trabaje con matriculas españolas. Este no va a ser el código final del proyecto, pero lo he utilizado
    para aprender opencv principalmente. Voy a tomar un camino diferente para el código final, basandome en Feautre Maching y Homography.
    con opencv.
     
    También quiero adaptar el proyecto a que trabaje con video en directo mediante camaras IP que es para lo que estaba destinado en un principio.    
"""

import cv2
import pytesseract
import re
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Preprocess image: grayscale, blur, edge detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

def detect_plate(edges, original_image):
    """Detect plate region using contours."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            return original_image[y:y+h, x:x+w]
    return None

def recognize_text(plate_image):
    """Recognize text from plate image using Tesseract."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 11'
    text = pytesseract.image_to_string(thresh, config=custom_config, lang='spa')
    return text.strip()

def validate_plate_format(text):
    """
    Validates and corrects the license plate format.
    Returns (final_text, message)
    """
    # Remove spaces and non-alphanumeric for matching, but keep original for display
    cleaned = re.sub(r'[^A-Z0-9 ]', '', text.upper())
    # Match 0000 AAA
    match = re.match(r'^(\d{4})\s*([A-Z]{3})$', cleaned)
    if match:
        return f"{match.group(1)} {match.group(2)}", None
    # Match A0000 AAA (leading letter)
    match_leading = re.match(r'^[A-Z](\d{4})\s*([A-Z]{3})$', cleaned)
    if match_leading:
        return f"{match_leading.group(1)} {match_leading.group(2)}", "Leading letter detected and removed."
    return cleaned, "Format does not match expected pattern."

class ANPR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.edges = None
        self.plate_image = None
        self.detected_text = ""
        self.final_text = ""
        self.format_message = None

    def preprocess(self):
        self.edges = preprocess_image(self.image_path)

    def detect(self):
        self.plate_image = detect_plate(self.edges, self.original_image)

    def recognize(self):
        if self.plate_image is not None:
            self.detected_text = recognize_text(self.plate_image)
            self.final_text, self.format_message = validate_plate_format(self.detected_text)
        else:
            self.detected_text = ""
            self.final_text = ""
            self.format_message = "No plate detected."

    def run(self):
        self.preprocess()
        self.detect()
        self.recognize()
        print("Detected License Plate Text:", self.detected_text)
        if self.format_message:
            print("Note:", self.format_message)
        print("Final Plate:", self.final_text)

def display_result(anpr_obj):
    if anpr_obj.plate_image is not None:
        plt.imshow(cv2.cvtColor(anpr_obj.plate_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Plate: {anpr_obj.final_text}")
        plt.axis('off')
        plt.show()
    else:
        print("No plate detected.")

if __name__ == "__main__":
    class_obj = ANPR("201-Imagenes/matricula4.jpg")
    class_obj.run()
    display_result(class_obj)
