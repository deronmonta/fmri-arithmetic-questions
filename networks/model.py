import torch
from torch import nn
import torchvision
#from networks.blocks import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    '''Two layer block for shallow resnet, for 18,34
    
    Arguments:
        nn {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    '''Three layer block for deeper resnet, used for 50, 101, 152
    
    Arguments:
        nn {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, seq_len ,num_classes=4):
        super(ResNet, self).__init__()
        # The sequence length is used as channel dimension
        self.in_planes = 64
        self.seq_len = seq_len

        self.conv1 = nn.Conv3d(seq_len, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(4096*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)

        # [batchsize ,channel, H, W, Z]
        return out


def ResNet18(seq_len):
    return ResNet(BasicBlock, [2,2,2,2],seq_len)

def ResNet34(seq_len):
    return ResNet(BasicBlock, [3,4,6,3],seq_len)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])




class Attention(nn.Module):
    """Attention network
    
    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, attention_dim,encoder_dim,decoder_dim):
        super(Attention,self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,encoded, decoder_hidden):
        """
        
        Arguments:
            encoded {} -- encoded volume, tensor of dimension [batchsize, num_pixels, encoder_dim]
            decoder_hidden {} -- 
        """

        attention1 = self.encoder_att(encoded)
        attention2 = self.decoder_att(decoder_hidden)

        full_attention = self.full_att(self.relu(attention1 + attention2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(full_attention)
        attention_weighted_encoding = (encoded * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding,alpha

class Decoder(nn.Module):

    def __init__(self,attention_dim,decoder_dim,seq_len,encoder_dim=512,num_task=5, dropout=0.5):
        """Decoder with attention
        
        Arguments:
            attention_dim {[type]} -- [description]
            decoder_dim {[type]} -- [description]
        
        Keyword Arguments:
            encoder_dim {int} -- [description] (default: {2048})
            num_task {int} -- total number of possible tasks  (default: {5})
        """
        super(Decoder,self).__init__()
        
        self.encoder_dim = encoder_dim 
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.num_task = num_task


        self.attention = Attention(self.attention_dim,self.encoder_dim,self.decoder_dim)
        self.decode_step = nn.LSTMCell(self.encoder_dim,self.decoder_dim,bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate        

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim,num_task)
        self.dropout = nn.Dropout(p=self.dropout)


        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        # Initialize LSTM state
        

        mean_encoder_out = encoder_out.mean(dim=1)
        print(mean_encoder_out.shape)
        h = self.init_h(mean_encoder_out)  # hidden state (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out) # cell state
        
        return h,c

    def forward(self,encoded,decode_length):
        """Forward propagation
        
        Arguments:
            encoded {} -- encoded image 
            decode_length {} -- 
        """

        batch_size = encoded.size(0) # 1st dimension is the batch size
        encoder_dim = encoded.size(1)
        num_pixels = encoded.size(2)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        #decode_lengths = (decode_length - 1).tolist()
        decode_lengths = list(range(0,decode_length))
        



        encoded = encoded.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoded)  # (batch_size, decoder_dim)

        
         # Create empty tensors to hold prediction and alphas
        predictions = torch.zeros(batch_size, decode_length, self.num_task).to(device)
        alphas = torch.zeros(batch_size, decode_length, 64).to(device)
        num_pixels = encoded.size(1)
        print('Decode lengths {}'.format(decode_lengths))
        for t in range(0,decode_length):
            print(t)
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoded,h)

        
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            

            h,c = self.decode_step(
                attention_weighted_encoding, (h, c)
                
            ) # (batchsize, decoder_dim)


            #preds = self.fc(self.dropout(h)) #(batchsize num-of-task)
            preds = self.fc(h)

            predictions[:, t-1, :] = preds
            alphas[:, t-1, :] = alpha

            #prediction: [batchsize, seq_length, num_task]

            return predictions ,decode_lengths,alphas

    

        




# class Encoder(nn.Module):
#     """[summary]
    
#     Arguments:
#         nn {[type]} -- [description]
#     """
#     def __init__(self,num_in_channel,num_filter):
#         super(Encoder,self).__init__()
#         self.num_in_channel = num_in_channel
#         #self.num_out_channel = num_out_channel
#         #self.num_filters = num_filters

#         self.block1 = ResidualBlock(in_channels=num_in_channel,out_channel=num_filter*4)
#         self.block2 = ResidualBlock(in_channels=num_filter*4,out_channel=num_filter*16)
#         self.block3 = ResidualBlock(in_channels=num_filter*16,out_channel=num_filter*64)
#         self.block4 = ResidualBlock(in_channels=num_filter*64,out_channel=num_filter*256)


#     def forward(self,input):

#         x = self.block1(input)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)


#         return x