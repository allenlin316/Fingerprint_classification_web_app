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
    fingerprint_id = fingerprint_img.split('__')[0]
    print(fingerprint_id)
    groundtruth_img = os.path.join(" ", "static", "Image", "fingerprints", fingerprint_img)
    pred = predict(fingerprint_img)
    pre_trained = request.form["methods"]
    if fingerprint_id != pred:
        split_txt = groundtruth_img.split("_")
        print(split_txt)
        pred_id = split_txt[0].rsplit('/', 1)[0] + '/' + str(pred)
        pred_img = pred_id + "__" + split_txt[2] + "_" + split_txt[3] + "_" + split_txt[4] + "_" + split_txt[5]
        print(pred_img)
        if not os.path.exists(pred_img): # 照片不存在
            if 'M' in pred_img:
                pred_img = pred_img.replace('__M_', '__F_')
            else:
                pred_img = pred_img.replace('__F_', '__M_')
            print(f'img not exist so change it into new path {pred_img}')
        correct = "False"
    else:
        pred_img = groundtruth_img
        correct = "True"
    return render_template('classification.html', groundtruth_id=fingerprint_id, groundtruth_img=groundtruth_img, pred_img=pred_img, prediction_id=pred, pre_trained=pre_trained, correct=correct)