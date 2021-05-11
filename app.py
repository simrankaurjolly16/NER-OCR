import os
from flask import Flask, render_template, request
import re
import csv
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import spacy
import pandas as pd


__file__ = 'D:\\My_learning\\CascadeTabNet\\Demo\\img'

app = Flask(__name__, template_folder='D:\\My_learning\\CascadeTabNet\\Demo\\ocr-app\\templates\\')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['GET','POST'])
def upload():
    target = os.path.join(__file__, 'images/')
#     print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
#         print(file)
        filename = file.filename
        destination = "/".join([target, filename])
#         print(destination)
        file.save(destination)
    
#         image = destinationpython app.py
        image = cv2.imread(destination)
        print("1")
        
        img = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        kernel_size=(5, 5)
        sigma=1.0
        amount=1.0 
        threshold=0
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)
        
        gray_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # result = sharpened.copy()
        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(sharpened, [c], -1, (255,255,255), 5)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(sharpened, [c], -1, (255,255,255), 5)
        
        kernel = np.ones((1, 1), np.uint8)
        gray_image = cv2.dilate(sharpened, kernel, iterations=1)
        gray_image = cv2.erode(gray_image, kernel, iterations=1)
        
        tesseract_config = r'--oem 1 --psm 6'
        details = pytesseract.image_to_string(gray_image, output_type=pytesseract.Output.STRING,
                                        config=tesseract_config, lang='eng')
        
        nlp = spacy.load('en_core_web_lg')
        doc = nlp(details)
        
        table = []
        for ent in doc.ents:
            table.append([ent.text,ent.label_,spacy.explain(ent.label_)])
        
        df2 = pd.DataFrame(table, columns=['Entity', 'Label','Label_description']).sort_values(by=['Label'])
        
        df3= df2[df2['Label'].isin(['PERSON'])]
        df4=df3.drop_duplicates(subset=None, keep='first',inplace= False)
        
        df4.to_csv('D:\\My_learning\\CascadeTabNet\\Demo\\ocr-app\\templates\\data.csv', index=False)
        print(df4)
        df4.to_html('D:\\My_learning\\CascadeTabNet\\Demo\\ocr-app\\templates\\table1.html')
        print("success")
        

    return render_template("table1.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=4555)
#     app.run(port=4555, debug=True)



