import os
import cv2
import numpy as np
from model.Unet import Unet
import torch
from torch import nn
from dataloader import Dataset, gen_test_img
from torch.optim import Adam
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import models

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resume = True

dataset = CustomDataset()
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

unet = Unet().to(device)

resume_epoch = 0

# if resume:
#     unet = torch.load("./checkpoints/epoch_90.pth")
#     resume_epoch = 90

# print(unet)

optimizer = Adam(unet.parameters(), lr=LR, betas=(0.5, 0.999))
loss = nn.MSELoss()
test_input = gen_test_img().to(device)


for epoch in range(resume_epoch, EPOCHS):
    for i, (input, label) in enumerate(trainloader):
        unet.train()
        optimizer.zero_grad()
        input = input.to(device)
        label = label.to(device)

        output = unet(input)

        error = loss(output, label)*2
        error.backward()
        optimizer.step()

        print(f"Epoch {epoch} step {i}: MSELoss: {error}\n")
        if i % 100 == 0:
            print('saving the output')
            unet.eval()
            vutils.save_image(unet(test_input),'results/output_epoch_%03d.png' % (epoch),normalize=True)
    if epoch % 10 == 0 and epoch != 0:
        torch.save(unet, f"./checkpoints/epoch_{epoch}.pth")
torch.save(unet, f"./checkpoints/final.pth")
