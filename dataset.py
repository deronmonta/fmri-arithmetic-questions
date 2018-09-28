import torch
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
import numpy as np
import pickle
import utils

class FMRI_Dataset(Dataset):
    """FMRI data set, the dimensions are 61 X 73 X 61 [X, Y ,Z]
    
    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.pat_lis = os.listdir(self.data_dir)
        
        for patient in self.pat_lis:
            

    def __getitem__(self, index):


        return
    
    def __len__(self):
            return 