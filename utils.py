import nibabel as nib
import os
import numpy as np
import pandas as pd
from operator import truediv
from scipy import ndimage, misc
import random

def load_hdr(filename):
    """Load an .hdr file
    
    Arguments:
        filename {str} -- path to .hdr file
    
    Returns:
        [np_arr] -- numpy array
    """

    img = nib.load(filename)
    np_arr = img.get_data()
    
    return np_arr

def normalize(volume):
    """Clip a fmri volume to 0~256
    
    Arguments:
        volume {[np array]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    MIN_BOUND = 0
    MAX_BOUND = 256.0
    volume = (volume - MIN_BOUND) /(MAX_BOUND - MIN_BOUND)
    volume[volume > 1] = 1 #Clip everything larger than 1 and 0
    volume[volume < 0] = 0
    volume = (volume*255).astype('uint8')

    return volume


def rescale_volume(volume, old_dimension, new_dimension,target_dimension):
    '''
    Rescale a volume according to new dimension
    Args:
        volume: 
        old_dimension: []
        new_dimension: [] 
        order: order of the spline interpolation
    '''
    old_dimension = volume.shape
    resize_factor = [x/y for x, y in zip(target_dimension,old_dimension)]
    rescaled_volume = ndimage.zoom(volume, resize_factor, mode = 'nearest')

    return rescaled_volume

def get_dataframe(data_dir):
    subject_id_lis = []
    full_path_lis = []
    for patient_id in os.listdir(data_dir):

        subject_id_lis.append(patient_id)

        #need to go in another subdirectory
        full_path = os.path.join(data_dir,patient_id)
        #full_path = os.path.join(full_path,patient_id)
        
        
        full_path_lis.append(full_path)
        # print(full_path_lis)


    df = pd.DataFrame({'patient_id':subject_id_lis, 'full_path':full_path_lis})

    return df


def get_hdr(patient_dir,filenames,get_single=False):
    """return a list of all numpy volumes in a patient directory
    
    Arguments:
        patient_dir {str} -- []
        start_index {int} -- [starting index for a sequence]
        get_single {bool} -- select single volume from a patient, for testing only
    
    Returns:
        hdr_lis [numpy array] -- [numpy array with dimension [sequence_length, H, W , Z]]
    """

    hdr_lis = []

    if  get_single:

        for filename in os.listdir(patient_dir):
            if filename.endswith('.hdr'):


                print(filename)
                hdr = load_hdr(os.path.join(patient_dir,filename))
                hdr = rescale_volume(hdr,old_dimension=[61,73,61],new_dimension=(64,64,64),target_dimension=[64,64,64])
                hdr = normalize(hdr)
                print(hdr.shape)
                hdr_lis.append(hdr)

        return hdr_lis


    else:
        hdr_lis = []
        hdr_stacked = np.empty((len(filenames),64,64,64))
        for row in filenames:
            print(row)
            # hdr_filename = row['filename']
            hdr = load_hdr(os.path.join(patient_dir,row))
            hdr = rescale_volume(hdr,old_dimension=[61,73,61],new_dimension=(64,64,64),target_dimension=[64,64,64])
            hdr = normalize(hdr)
            hdr_lis.append(hdr)
        print(len(hdr_lis))
        for index, hdr in enumerate(hdr_lis):
            hdr_stacked[index,:,:,:] = hdr

        print('Stacked hdr shape {}'.format(hdr_stacked.shape))

        return hdr_stacked


def get_sequence(seq_df, window_size):
    """Get a sequence 
    
    Arguments:
        seq_df {} -- 
        window_size {} -- 
    
    Returns:
        start_idx -- starting index of the sequence 
        seq -- 
        filenames {pd}
    """

    start_idx = random.randint(0,270-window_size)
    #seq = seq_df.loc[start_idx:start_idx+window_size-1,'task_id']
    filenames = seq_df.loc[start_idx:start_idx+window_size-1,'filename']
    #print('file name length {}'.format(len(filenames)))
    seq = seq_df.loc[start_idx:start_idx+window_size-1,'task_id'].values

    return start_idx, seq, filenames


        
        
