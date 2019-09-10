import pickle
import pandas as pd
import json
import torch
import numpy as np


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

        unwanted = ['ID', 'copy_num','chi2SALT2', 'chi2sGauss', 'chi2Salt2']
        
        self.data = self.data.drop(unwanted,axis=1)

        print('Columns to be used:')
        print(list(self.data.columns))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = np.array((self.data.iloc[idx][:-1].values), dtype=float)

        y = int(self.data.iloc[idx]['target'])
        sample = {'features': x, 'label': y}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def input_size(self):
        return len(self.data.columns)-1


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample['features'], sample['label']

        return {'features': torch.from_numpy(x),
                'label': torch.tensor(y)}
