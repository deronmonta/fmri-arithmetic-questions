import torch
from torch import nn
import torchvision

class CNN(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
def __init__(self,num_in_channel,num_out_channel,num_filters):
    super(CNN,self).__init__()
    self.num_in_channel = num_in_channel
    self.num_out_channel = num_out_channel
    self.num_filters = num_filters

