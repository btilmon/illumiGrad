import time
import torch
import numpy as np
from options import Options
from camera import Camera
from dataset import CustomDataset
import sys, os, subprocess, glob
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.weight"] = "bold"
matplotlib.rcParams.update({'font.size': 12})

class Main():
    def __init__(self, opt):
        self.opt = opt
        self.camera = Camera(opt)
        self.optimizer = torch.optim.Adam(self.camera.parameters())
        self.dataset = CustomDataset(self.opt)
        self.dataLoader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.lossFunction = torch.nn.MSELoss()

    def run(self):
        # load initial camera pair
        self.data = next(iter(self.dataLoader))
        color = self.data[("color", 0)]
        depth = self.data[("depth", 0)]
        mask = self.data[("mask", 0)]
        beforecolorE = self.camera.colorE.clone()
        beforedepthK = self.camera.depthK.clone()
        initNum = 200
        # initialize training on single pair
        print("\n Begin optimizing on initial color-depth pair in trajectory \n")
        for i in range(initNum):
            self.optimizer.zero_grad()
            predColor = self.camera(depth, color)
            loss = self.lossFunction(predColor[mask], color[mask])
            loss.backward()
            self.optimizer.step()
            print("loss:", loss)

            if i == 0: # save before calibration picture
                predColor[~mask] = color[~mask]
                depth = (depth - depth.min()) / depth.max()
                result = torch.cat((predColor, color, depth.repeat(1,3,1,1)), dim=3)
                before = result[0].permute(1,2,0).detach().numpy()

        # temporally optimize based on trajectory
        print("Continue optimizing throughout trajectory")
        for i in range(0, 50):
            color = self.data[("color", i)]
            depth = self.data[("depth", i)]
            mask = self.data[("mask", i)]

            self.optimizer.zero_grad()
            predColor = self.camera(depth, color)
            loss = self.lossFunction(predColor[mask], color[mask])
            loss.backward()
            self.optimizer.step()
            print("loss:", loss)
            
        # post-calibration result
        predColor[~mask] = color[~mask]
        depth = (depth - depth.min()) / depth.max()
        result = torch.cat((predColor, color, depth.repeat(1,3,1,1)), dim=3)
        result = result[0].permute(1,2,0).detach().numpy()
        plt.subplot(211)
        plt.title("Before Calibration")
        plt.imshow(before)
        plt.axis("off")
        plt.subplot(212)
        plt.title("After Calibration")
        plt.imshow(result)
        plt.axis("off")
        plt.show()

        print("EXTRINSICS")
        print(beforecolorE)
        print(self.camera.colorE)
        print("\n INTRINSICS")
        print(beforedepthK)
        print(self.camera.depthK)
         
if __name__ == "__main__":
    opt = Options()
    opt = opt.parse()
    main = Main(opt)
    main.run()
