from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from  typing import Any
from Config.config import *
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
class ImageSet(Dataset):
    def __init__(self, img_path, label_path, mean, std, mean_std_static, transform=None):
        super().__init__()
        datasets = np.load(img_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)
        if not mean_std_static:
            mean = np.mean(datasets, axis=(0, 1, 2))
            std = np.std(datasets, axis=(0, 1, 2))
        datasets = (datasets - mean) / std
        self.datasets = datasets.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.datasets[index] 
        label = self.labels[index]
        
        
        num_points = img.shape[0] * img.shape[1]
        img = img.reshape(num_points, -1)  
        if self.transform is not None:
            img = self.transform(img)
        
     
        img = torch.from_numpy(img)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    def __len__(self):
        return len(self.datasets)
def data_loader(img_path, label_path, mean=0.0, std=1.0, mean_std_static=True, batch_size=32, shuffle=False, num_workers=0):
    
    dataset_train = ImageSet(img_path, label_path, mean, std, mean_std_static, transform=None)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return loader_train
if __name__ == "__main__":
    
    img_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Train/imgs.npy'
    label_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Train/labels.npy'
    
    loader = data_loader(img_path, label_path, num_workers=0, mean_std_static=True)
    for i, (img, label) in enumerate(loader):
        print(f'Batch {i}: img shape: {img.shape}, label shape: {label.shape}')
        if i == 0:
            break

    
