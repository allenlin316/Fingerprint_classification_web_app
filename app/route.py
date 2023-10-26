from flask import request, render_template
import os , re
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
    print(f'fingerprint_id: {fingerprint_id}')
    groundtruth_img = os.path.join("/static", "Image", "fingerprints", fingerprint_img)
    print(f'groundtruth_img: {groundtruth_img}')
    pred = predict(fingerprint_img)
    pre_trained = request.form["methods"]
    if int(fingerprint_id) != int(pred):
        pred_img_path = re.sub(r"\d+", str(pred), groundtruth_img, 1) # Use regular expression to replace the number
        #print(f'os.getcwd(): {os.getcwd()}')
        #print(f'pred_img_path: {os.getcwd() + "/app" + pred_img_path}')
        #pred_img_path = os.getcwd() + "/app" + pred_img_path
        print(f'pred_img_fullpath: {pred_img_path}')
        if not os.path.exists(os.getcwd() + "/app" + pred_img_path): # 照片不存在
            if 'M' in pred_img_path:
                pred_img_path = pred_img_path.replace('__M_', '__F_')
            else:
                pred_img_path = pred_img_path.replace('__F_', '__M_')
            print(f'img not exist so change it into new path {pred_img_path}')
        correct = "False"
    else:
        pred_img_path = groundtruth_img
        correct = "True"
    print(f'pred_img: {pred_img_path}')
    return render_template('classification.html', groundtruth_id=fingerprint_id, groundtruth_img=groundtruth_img, pred_img=pred_img_path, prediction_id=pred, pre_trained=pre_trained, correct=correct)