import os
import cv2
import numpy as np
from model.Unet import Unet
from torchvision.transforms import transforms
import torchvision.utils as vutils.

gray_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet = Unet().to(device)
unet.load_state_dict(torch.load("checkpoints/final.pth"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input = gray_transform(gray)
    input = input.to(device)
    with torch.no_grad():
        output = unet(input)
