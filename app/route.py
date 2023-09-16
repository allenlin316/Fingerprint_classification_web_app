from flask import request, render_template
import os 
import sys
sys.path.append("./prediction")
from prediction import predict


def hello_world():
    return "Hello, 這是使用 MVC 框架建的網站為嘉義大學畢業專題"

def index():
    return render_template('index.html') 

def classification():
    return render_template('classification.html')

def results():
    fingerprint_img = request.files['file'].filename
    img = os.path.join(" ", "static", "Image", fingerprint_img)
    pred = predict(fingerprint_img)
    return render_template('classification.html', img=img, prediction_id=pred)    
