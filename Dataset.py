from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class Dataset(TensorDataset):
    def __init__(self, path, size=512):
        self.path = path
        self.size = size
        self.datasets = os.listdir(path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        high = Image.open(self.path + self.datasets[item]).resize([self.size, self.size])
        low = Image.open(self.path + self.datasets[item]).resize([256, 256])
        return self.transforms(low), self.transforms(high)

    def __len__(self):
        return len(self.datasets)

class Dataset_svm(TensorDataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        v = np.float32(self.data[item])
        y = -1 if int(self.label[item].strip().split()[1]) == 0 else 1
        return v, y

    def __len__(self):
        return self.data.shape[0]