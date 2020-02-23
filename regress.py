from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#in_channel, out_channel, kernel_size, stride, padding=0 (default)
		self.conv1 = nn.Conv2d(1, 16, 5, 1) 
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(16, 16, 5, 1)
		self.bn2 = nn.BatchNorm2d(16)

		self.conv3 = nn.Conv2d(16, 64, 2, 1) 
		self.bn3 = nn.BatchNorm2d(64)

		self.conv4 = nn.Conv2d(64, 128, 2, 1)
		self.bn4 = nn.BatchNorm2d(128)

		self.dropout1 = nn.Dropout2d(0.3)
		self.dropout2 = nn.Dropout2d(0.3)
		self.dropout3 = nn.Dropout2d(0.3)

		self.fc1 = nn.Linear(4608, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 32)
		self.fc4 = nn.Linear(32, 1)

	def forward(self, x):

		h = F.relu(self.conv1(x))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn1(h)

		h = F.relu(self.conv2(h))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn2(h)

		h = F.relu(self.conv3(h))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn3(h)

		h = F.relu(self.conv4(h))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn4(h)

		h = torch.flatten(h, start_dim=1) #flatten input of [bs, c, w, h], so from dim=1
		h = F.relu(self.fc1(h))
		h = self.dropout1(h)
		h = F.relu(self.fc2(h))
		h = self.dropout2(h)
		h = F.relu(self.fc3(h))
		h = self.dropout3(h)
		h = F.relu(self.fc4(h))

		h = h.squeeze() #squeezing to reduce dims from (64,1) to (64,) to match target
		output = h
		return output

#joshua's model
class Net2(nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		#in_channel, out_channel, kernel_size, stride, padding=0 (default)

		self.conv1 = nn.Conv2d(1, 16, 7, 1) 
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(16, 16, 5, 1)
		self.bn2 = nn.BatchNorm2d(16)

		self.dropout1 = nn.Dropout2d(0.3)
		self.dropout2 = nn.Dropout2d(0.3)
		self.dropout3 = nn.Dropout2d(0.3)

		self.fc1 = nn.Linear(12544, 512)
		self.fc2 = nn.Linear(512, 1)

	def forward(self, x):

		h = F.relu(self.conv1(x))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn1(h)

		h = F.relu(self.conv2(h))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn2(h)

		h = torch.flatten(h, start_dim=1) #flatten input of [bs, c, w, h], so from dim=1
		h = F.relu(self.fc1(h))
		h = self.dropout1(h)
		h = F.relu(self.fc2(h))
		h = h.squeeze() #squeezing to reduce dims from (64,1) to (64,) to match target
		output = h
		return output

def test(x):
	
	device = torch.device("cuda")
	model = Net()
	#if load using state_dict, everythin becomes 1
	model = torch.load('regressor.pt')
	model = Net().to(device)
	model.eval()
	J = model(x)

	#test without network
	# J = x.mean(dim=(2,3))
	# J.squeeze(1)
	return J
