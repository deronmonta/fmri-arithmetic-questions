import nibabel as nib
import os
import numpy as np

def load_hdr(filename):

    img = nib.load(filename)
    np_arr = img.get_fdata()
    
    return np_arr