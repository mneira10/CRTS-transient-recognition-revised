import pickle
import pandas as pd
import json
import torch
import numpy as np
from torchvision import transforms
import pdb


class dataSet():

    def __init__(self, dataset, transform=None):
        with open('config.json') as f:
            config = json.load(f)

        dataPath = config['dataPath']

        self.data = pickle.load(open(dataPath+dataset+'.pkl', "rb"))
        self.data = self.data.sample(frac=1)

        self.data = self.data.drop('Class', axis=1)
        self.data = self.data.drop('ObsCount', axis=1)
        self.data = self.data.reset_index()

        print(self.data.columns)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = np.array((self.data.iloc[idx][2:-1].values), dtype=float)

        y = int(self.data.iloc[idx]['target'])

        sample = {'features': x, 'label': y}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def input_size(self):
        return len(self.data.columns)-3


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample['features'], sample['label']

        return {'features': torch.from_numpy(x),
                'label': torch.tensor(y)}
