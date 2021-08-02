import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ImgsDataset(Dataset):
    def __init__(self, path_to_pkl:str, train:bool=True):
        self.train = train
        data = np.load(path_to_pkl, allow_pickle=True)
        
        if self.train:
            self.imgs = data['train'].transpose(0,3,1,2).astype(float)
            self.labels = data['train_labels']
        else:
            self.imgs = data['test'].transpose(0,3,1,2).astype(float)
            self.labels = data['test_labels']
        
        if "colored" not in path_to_pkl:
            self.imgs /= 255

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

def get_dataloaders(dataset_name:str, batch_size:int):
    assert dataset_name in ['shapes','shapes_colored','mnist','mnist_colored']

    dataset_loc = "/content/RealNVP/data/{}.pkl".format(dataset_name)
    train_dataset = ImgsDataset(path_to_pkl=dataset_loc, train=True)
    test_dataset = ImgsDataset(path_to_pkl=dataset_loc, train=False)
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    return train_loader, test_loader
