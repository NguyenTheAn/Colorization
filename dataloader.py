import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import random

def gen_test_img():
    l = np.load("./data/50512_125589_bundle_archive/l/gray_scale.npy")
    gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    for i in range(0, 63):
        index = random.randint(0, len(l-1))
        gray = l[index]
        gray = gray_transform(gray)
        gray = torch.unsqueeze(gray, 0)
        if i == 0:
            input = gray
        input = torch.cat((input, gray), 0)

    return input


class CustomDataset(data.Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        self.ab = np.load("./data/50512_125589_bundle_archive/ab/ab/ab1.npy")
        self.l = np.load("./data/50512_125589_bundle_archive/l/gray_scale.npy")

    def transform(self, gray, image):
        gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if random.random() > 0.5:
            gray = TF.hflip(gray)
            image = TF.hflip(image)
        if random.random() > 0.5:
            gray = TF.rotate(gray, 30)
            image = TF.rotate(image, 30)
        if random.random() > 0.5:
            gray = TF.vflip(gray)
            image = TF.vflip(image)
        if random.random() > 0.5:
            affine = transforms.Compose([
                transforms.RandomAffine(30)
            ])
            gray = affine(gray)
            image = affine(image)

        gray = gray_transform(gray)
        image = normal_transform(image)
        return gray, image

    def __getitem__(self, index):

        index = index % len(self.ab)
        img = np.zeros((224, 224, 3))
        gray = self.l[index]
        img[:, :, 0] = gray
        img[:, :, 1:] = self.ab[index]
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        gray = Image.fromarray(np.uint8(gray), 'L')
        img = Image.fromarray(img)
        gray, img = self.transform(gray, img)
        
        return gray, img
    
    def __len__(self):
        return len(self.ab) * 3
