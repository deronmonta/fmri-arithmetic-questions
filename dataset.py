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


        The sequence encoding are as follows:
        0: Blank
        1: Large addition
        2: Small additions
        3: Large multiplication
        4: Small multiplication
    
    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(self,data_dir,seq_csv,seq_len):
        self.data_dir = data_dir
        self.df = get_dataframe(data_dir) # patient id dataframe
        self.seq_df = pd.read_csv(seq_csv,header=0)
        self.seq_len = seq_len

        print(self.df)
        print(self.seq_df)

            

    def __getitem__(self, index):

        data_dir = self.df.loc[index,'full_path']

        start_index, seq, filenames = get_sequence(seq_df=self.seq_df, window_size=self.seq_len)
        print(filenames)
        volumes = get_hdr(data_dir,filenames=filenames,get_single=False)
        

        #volumes = torch.FloatTensor(volumes).cuda()


        # print(data_dir)
        # print(seq)


        sample = {'volumes':volumes,'sequence':seq}

        return sample
    
    def __len__(self):
            return len(self.df)
