import cv2
import pytesseract
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

class OCREngine:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def deskew_and_clean(self, pil_image):
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        try:
            osd = pytesseract.image_to_osd(img)
            angle = re.search('(?<=Rotate: )\d+', osd).group(0)
            if angle == '90': img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == '180': img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == '270': img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except: pass 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def extract_text(self, file_path):
        """Processes file and returns CLEANED text for NLP."""
        combined_text = []
        
        if file_path.lower().endswith('.pdf'):
            pages = convert_from_path(file_path, 300)
            for page in pages:
                clean_page = self.deskew_and_clean(page)
                text = pytesseract.image_to_string(clean_page, lang='fra')
                combined_text.append(text)
        else:
            img = Image.open(file_path)
            clean_img = self.deskew_and_clean(img)
            text = pytesseract.image_to_string(clean_img, lang='fra')
            combined_text.append(text)

        raw_string = "\n".join(combined_text).strip()

        # NLP CLEANING: Remove noise so the model trains better
        # 1. Remove "--- Page X ---" if any
        clean_string = re.sub(r'--- Page \d+ ---', '', raw_string)
        # 2. Keep only letters, numbers, spaces, and identity markers like '<'
        clean_string = re.sub(r'[^\w\s<>]', ' ', clean_string)
        # 3. Lowercase and remove extra spaces
        return clean_string.lower().strip()
