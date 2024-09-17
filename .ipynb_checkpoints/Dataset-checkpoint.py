## Dataset 
import torch
from torch.utils.data.dataset import Dataset
import os
import matplotlib.pyplot as plt
import scipy.io
from torchvision import transforms
from Config import Config
import cv2
config = Config()

my_transform = transforms.Compose([
transforms.ToTensor(),    
transforms.Resize((config.IMG_SIZE))
])


class Dataset(Dataset):
    def __init__(self,data_path = "../Data/OCT2/",seg_path = "../Data/Segmentation/MANUAL_SEGMENTATIONS_KERMANY_DATASET/",mode = "autolabel" , transforms=my_transform):
        super().__init__()
        self.img = sorted(os.listdir(data_path))
        self.seg = sorted(os.listdir(seg_path))
        self.data_path , self.seg_path = data_path , seg_path
        self.transforms = transforms
        self.mode = mode
    def __len__(self):
        return min(len(self.img),len(self.seg))
    def __getitem__(self,index):
        if self.img[index].split(".")[0] != self.seg[index].split(".")[0]:
            return "Not Found !"
        else:
            self.rimg = cv2.imread(os.path.join(self.data_path ,self.img[index]),cv2.IMREAD_GRAYSCALE)
            self.rseg =  scipy.io.loadmat(os.path.join(self.seg_path , self.seg[index]))[self.mode]
            return self.transforms(self.rimg) , self.transforms(self.rseg) , self.img[index].split(".")[0]
    def __del__(self):
        del self.rimg
        del self.rseg