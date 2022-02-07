import os
from torch.utils.data.dataset import Dataset
import h5py
from skimage import io, transform
import torch
from PIL import Image


class syn_dataset(Dataset):
    def __init__(self, data_path, transform):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.img_filelist = []
        self.label_filelist = []
        self.transform = transform

        for i, cls in enumerate(self.clslist):
            file_list = os.listdir(os.path.join(self.filepath, cls))

            for file in file_list:
                self.img_filelist.append(os.path.join(self.filepath, cls, file))
                self.label_filelist.append(i)


        self.data_num = len(self.img_filelist)

    def __getitem__(self, index):
        file_pth = self.img_filelist[index]
        label = self.label_filelist[index]


        image = io.imread(file_pth)
        image = Image.fromarray(image)

        image = self.transform(image)

        return image, label


    def __len__(self):
        return self.data_num  #

