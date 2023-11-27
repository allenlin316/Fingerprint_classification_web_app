import os, cv2, torch
from PIL import Image
import json
import argparse
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import moco.builder
import hybrid_resnet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = None
data_transforms = None
img_dir = os.path.join('app', 'static', 'Image', 'fingerprints')
parser = argparse.ArgumentParser(description='Execute linear probe experiment')
# parser.add_argument('-d', '--datadir', metavar='DIR', default="data", type=Path,
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--classes', default=600, type=int,
                    help='Number of classes in the training set (default:10)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--sigtemp', default=1.0, type=float,
                    help='Pre-quantum Sigmoid temperature (default: 1.0)')

parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
                    help='If enabled, apply BatchNorm1d to the input of the pre-quantum Sigmoid.')

parser.add_argument('--identity', dest='identity', action='store_true',
                    help='If enabled, the test network is replaced by the identity. The previous and subsequent layer '
                         'still compress to n_qubits however.')
parser.add_argument('-w', '--width', type=int, default=4,
                    help='Width of the test network (default: 4). If quantum, this is the number of qubits.')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of layers in the test network (default: 2).')

parser.add_argument('-q', '--quantum', dest='quantum', action='store_true',
                    help='If enabled, use a minimised version of ResNet-18 with QNet as the final layer')
parser.add_argument('--q_backend', type=str, default='qasm_simulator',
                    help='Type of backend simulator to run quantum circuits on (default: qasm_simulator)')

parser.add_argument('--encoding', type=str, default='vector',
                    help='Data encoding method (default: vector)')
parser.add_argument('--q_ansatz', type=str, default='sim_circ_14_half',
                    help='Variational ansatz method (default: sim_circ_14_half)')
parser.add_argument('--q_sweeps', type=int, default=1,
                    help='Number of ansatz sweeeps.')
parser.add_argument('--activation', type=str, default='partial_measurement_half',
                    help='Quantum layer activation function type (default: partial_measurement_half)')
parser.add_argument('--shots', type=int, default=100,
                    help='Number of shots for quantum circuit evaulations.')
parser.add_argument('--save-dhs', action='store_true',
                    help='If enabled, compute the Hilbert-Schmidt distance of the quantum statevectors belonging to'
                         ' each class. Only works for -q and --classes 2.')

parser.add_argument('--submission-time', type=str, default='',
                    help='Date and time of yaspify submission to create output directory.')
args = parser.parse_args(args=['--gpu', '0', '-q', '--q_backend', 'qasm_simulator', '--q_ansatz', 'sim_circ_14_half', '-w', '8', '-a', 'resnet18'])
args.pretrained = os.path.join("app", "models", "hybrid_simclr.pth.tar") # pre-trained network

def load_model(model_path): # set pre-trained model
    global model
    # Load hyperparams of trained network
    train_args = json.load(open(os.path.join(os.path.dirname(args.pretrained), "train_args.json"), "r"))
    args.arch = train_args["arch"]
    args.identity = train_args["identity"]
    args.width = train_args["width"]
    args.layers = train_args["layers"]
    args.quantum = train_args["quantum"]
    args.q_ansatz = train_args["q_ansatz"]
    args.q_sweeps = train_args["q_sweeps"]
    args.activation = train_args["activation"]
    args.shots = train_args["shots"]
    model = moco.builder.SimCLR(hybrid_resnet.resnet18, args=args).encoder
    model.fc = torch.nn.Linear(model.fc[0].in_features, args.classes, bias=True)
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    print(f'hybrid model: {model}')
    
def load_data(filename):
    img_path = os.path.join(img_dir, filename)
    image = cv2.imread(img_path, 0) # load image with grayscale
    return image
    
def load_transform():
    global data_transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(), # to PIL format
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5071, 0.5071, 0.5071], std=[0.4107, 0.4107, 0.4107]), # image = (image-mean) / std
    ])
    
def predict_hybrid(fingerprint_img):
    model_path = args.pretrained
    load_model(model_path)
    load_transform()
    image = load_data(fingerprint_img)
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    print(f'image_tensor: {image_tensor.shape}')
    # evaluate model:
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        print(f'hybrid outputs: {outputs}')
        #pred = torch.argmax(outputs).item()
        pred = torch.mode(outputs.argmax(1)).values.item()
        print(f'pred: {pred}')
    return pred