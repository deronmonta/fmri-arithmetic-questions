import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
def __init__(self,num_in_channel,num_out_channel,num_filters):
    super(Encoder,self).__init__()
    self.num_in_channel = num_in_channel
    self.num_out_channel = num_out_channel
    self.num_filters = num_filters


class Attention(nn.Module):
    """"Attention network"
    
    Arguments:
        nn {[type]} -- [description]
    """
def __init__(self,encoder_dim,decoder_dim):



class Decoder(nn.Module):
    super(Decoder,self).__init__()

    self.decode_step = nn.LSTMCell()