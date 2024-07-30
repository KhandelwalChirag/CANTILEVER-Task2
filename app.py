from flask import Flask, request, render_template
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# For Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS/Linux
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def preprocess_image(image_path):

    # Open the image using PIL

    image = Image.open(image_path)
    
    # Convert to grayscale

    gray = image.convert('L')
    
    # Enhance contrast slightly

    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)
    
    # Apply a slight blur to smooth out the noise

    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    
    # Convert back to OpenCV format

    image_cv = np.array(gray)
    
    # Binarize the image 

    _, binary = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the processed image
    processed_image_path = os.path.join(UPLOAD_FOLDER, 'processed_image.png')
    cv2.imwrite(processed_image_path, binary)
    
    return processed_image_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        if file.filename == '':
            return "No selected file"
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:

                # Preprocess the image

                processed_image_path = preprocess_image(file_path)
                
                # Extract text using pytesseract

                text = pytesseract.image_to_string(processed_image_path)
                
                if not text.strip():
                    text = "No text detected in the image."

            except ValueError as e:
                return str(e)
            except pytesseract.pytesseract.TesseractError as e:
                return str(e)
            
            return render_template('result.html', text=text)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
