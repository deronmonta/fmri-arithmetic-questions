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

