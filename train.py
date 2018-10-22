from utils import *
from dataset import *


DATA_DIR = '/media/noahyang/Yale/fMRI_Data/fmri'
BATCH_SIZE = 2

fmri_dataset = FMRI_Dataset(DATA_DIR)
dataloader = DataLoader(fmri_dataset, batch_size=2)


for index, sample in enumerate(dataloader):

    print(sample)