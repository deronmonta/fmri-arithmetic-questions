import torch
from torch import nn
import torchvision
from networks.blocks import *
class Encoder(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self,num_in_channel,num_filter):
        super(Encoder,self).__init__()
        self.num_in_channel = num_in_channel
        #self.num_out_channel = num_out_channel
        #self.num_filters = num_filters

        self.block1 = ResidualBlock(in_channels=num_in_channel,out_channel=num_filter*4)
        self.block2 = ResidualBlock(in_channels=num_filter*4,out_channel=num_filter*16)
        self.block3 = ResidualBlock(in_channels=num_filter*16,out_channel=num_filter*64)
        self.block4 = ResidualBlock(in_channels=num_filter*64,out_channel=num_filter*256)


    def forward(self,input):

        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)


        return x






class Attention(nn.Module):
    """"Attention network"
    
    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, attention_dim,encoder_dim,decoder_dim):
        super(Attention,self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)





class Decoder(nn.Module):

    def __init__(self,attention_dim, ):
        super(Decoder,self).__init__()
        
        self.encoder_dim = encoder_dim
    

        #self.decode_step = nn.LSTMCell()