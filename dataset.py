import torch
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
import numpy as np
import pickle
from utils import *

class FMRI_Dataset(Dataset):
    """FMRI data set, the dimensions are 61 X 73 X 61 [X, Y ,Z] 
        There are 270 volumes for one subject, each volumes last two seconds accounting for 540 seconds. 

        Each block includes 14 real arithmetic equations and 2 false ones.
        
        First 2 seconds of each block are blank. 
        Each block 
    
    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.df = get_dataframe(data_dir)

        print(self.df)

            

    def __getitem__(self, index):

        data_dir = self.df.loc[index,'full_path']
        print(data_dir)
        volumes = get_hdr(data_dir,get_single=True)
        volumes = torch.FloatTensor(volumes).cuda()
        volumes = volumes[0]
        sample = {'volumes':volumes}

        return sample
    
    def __len__(self):
            return len(self.df)
