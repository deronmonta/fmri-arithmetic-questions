import nibabel as nib
import os
import numpy as np

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


def rescale_volume(volume, old_dimension, new_dimension, order):
    '''
    Rescale a volume according to new dimension
    Args:
        volume: 
        old_dimension: []
        new_dimension: [] 
        order: order of the spline interpolation
    '''
    target_shape = np.round(volume.shape * old_dimension / new_dimension)
    true_spacing = old_dimension * volume.shape / target_shape
    resize_factor = target_shape / volume.shape
    rescaled_volume = zoom(volume, resize_factor, mode = 'nearest',order=order)

    return rescaled_volume,true_spacing

def get_dataframe(data_dir):
    