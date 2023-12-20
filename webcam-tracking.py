import cv2
import torch
from model import load_model

# apply and load the model
model = load_model

# allow access to the webcam
cap = cv2.VideoCapture(0)

