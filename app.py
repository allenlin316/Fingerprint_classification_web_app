from flask import Flask, request, render_template
import pickle, os, cv2, torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import numpy as np

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = None
data_transforms = None
img_dir = os.path.join("static", "Image")

# using ResNet50 pre-trained model
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.model = models.resnet50()

        # 鎖定 ResNet18 預訓練模型參數
        # for param in self.model.parameters():
        #   param.requires_grad = False
 
        # 修改輸出層輸出數量
        self.model.fc = nn.Linear(2048, 600)
         # Modify the output layer

    def forward(self, x):
        x = self.model(x)
        return x

def load_model(model_path): # set pre-trained model
    global model
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

def load_data(filename):
    img_path = os.path.join(img_dir, filename)
    image = cv2.imread(img_path)
    return image
    
def load_transform():
    global data_transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(), # to PIL format
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.5071, 0.5071], std=[0.4107, 0.4107, 0.4107]), # image = (image-mean) / std
])

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fingerprint = request.files['file'].filename
    image = load_data(fingerprint)
    print(image.shape)
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    print(image_tensor)
    outputs = model(image_tensor)
    preds = outputs.argmax(1).item()
    print(preds)
    return render_template('index.html', img=os.path.join(img_dir, fingerprint), prediction_id=preds)


if __name__ == "__main__":
    model_path = os.path.join("models", "supervised_fingerprint_model.pt")
    load_model(model_path)
    load_transform()
    app.run(debug=True, host='0.0.0.0', port=5000)
