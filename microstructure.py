"""
Matsci data loader
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class MicrostructureDataset(Dataset):
    """
    Class to read the numpy dataset for the microstructure
    """
    def __init__(self, data_path, mode, transform=None):
        #self.data = h5py.File(data_path, mode='r')['all_morph']
        self.tuple_data = np.load(data_path, allow_pickle=True)
        self.data = self.tuple_data[:,0]
        self.label = self.tuple_data[:,1] # for J 
        self.mode = mode
        if self.mode == 'JF':
            self.ff = self.tuple_data[:,2] # for ff

        self.transform = transform

    #only return images
    # def __getitem__(self, index):

    #     #x = torch.FloatTensor(self.data[..., index]).unsqueeze(0)
    #     x = torch.FloatTensor(np.float32(self.data[index]))
    #     if self.transform is not None:
    #         x = self.transform(x)
    #     #print(x.min(), x.max())
    #     #p1 = torch.FloatTensor(x.mean())
    #     return x

    #return image and labels
    def __getitem__(self, index):

        #x = torch.FloatTensor(self.data[..., index]).unsqueeze(0)
        x = torch.FloatTensor(np.float32(self.data[index]))
        if self.transform is not None:
            x = self.transform(x)

        if self.mode == 'J':
            y = torch.FloatTensor(np.expand_dims(np.float32(self.label[index]),axis=0))
            return x, y

        elif self.mode == 'JF':
            y = torch.FloatTensor(np.expand_dims(np.float32(self.label[index]),axis=0))
            z = torch.FloatTensor(np.expand_dims(np.float32(self.ff[index]),axis=0))
            return x, y, z

        else:
            print('Please specify J or JF')
    
    def __len__(self):
        return self.data.shape[0]