from utils import *
from dataset import *
from networks.model import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn.utils.clip_grad import *
from torch import nn


#DATA_DIR = '/media/noahyang/Yale/fMRI_Data/fmri'

# Hyperparameters and configurations

DATA_DIR = './data'
BATCH_SIZE = 4
SEQ_CSV = './sequence.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#CRITERION = nn.CrossEntropyLoss().to(DEVICE)
CRITERION = nn.CrossEntropyLoss()
NUM_EPOCHS = 20
ENCODER_LR = 0.000001  # learning rate for encoder if fine-tuning
DECODER_LR = 4e-4     # learning rate for decoder

ATTENTION_DIM = 512
DECODER_DIM = 512

NUM_TASK = 5 # Number of different types of class
SEQ_LEN = 1 #Length of prediction duration

fmri_dataset = FMRI_Dataset(DATA_DIR,SEQ_CSV,SEQ_LEN)
dataloader = DataLoader(fmri_dataset, batch_size=BATCH_SIZE)


cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoder = nn.DataParallel(ResNet34(channel=1,num_classes=NUM_TASK, no_lstm=True)).cuda() # Dimension are [ batchsize , # of channel, H, W, Z]
decoder = Decoder(ATTENTION_DIM,DECODER_DIM,seq_len=SEQ_LEN)
#decoder_optimizer = torch.optim.Adam(decoder.parameters,lr=DECODER_LR)
encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=ENCODER_LR,amsgrad=True) 



# encoder = encoder.to(DEVICE)
#decoder = decoder.to(DEVICE)
print(encoder)
#print(decoder)


def train(dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer=None, epoch=20):
    '''Single epoch training
    
    Args:
            dataloader ([type]): [description]
            encoder ([type]): [description]
            decoder ([type]): [description]
            criterion ([type]): [description]
            encoder_optimizer ([type]): [description]
            decoder_optimizer ([type]): [description]
            epoch ([type]): [description]
    '''

    encoder.train()
    #decoder.train()

    for index, sample in enumerate(dataloader):


        encoder_optimizer.zero_grad()
        volumes = sample['volumes']
        seq = sample['sequence']

        # seq = class2onehot(seq,SEQ_LEN,BATCH_SIZE,NUM_TASK)
        # print('One hot shape {}'.format(seq.shape))



        volumes, seq = volumes.cuda(), seq.cuda()
        volumes,seq = volumes.float(), seq.long()
        

        # print('Volumes shape {}'.format(volumes.shape)) #[batchsize, seqlen, 64, 64, 64]
        # print('Sequence {}'.format(seq)) #[batchsize, seqlen]

        encoded_lis = []

        for seq_index in range(SEQ_LEN):

            #decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()

            one_volume = volumes[:,seq_index,:,:,:]
            # Need to unsqueeze TWICE

            one_volume = one_volume.unsqueeze(1)
            #[bs , 1 , 64 ,64 ,64]
            
            print('Single volume shape {}'.format(one_volume.shape))
            #[bs , 1 , 64 ,64 ,64]

            encoded = encoder(one_volume) # Single volume at sequence index 
            #scores, alpha = decoder(encoded,decode_length=1)
            #scores = encoded

            #scores = scores.float()
            #print('Scores: {}'.format(scores))
            #scores = scores.permute((0,2,1)) 
            
            encoder_optimizer.zero_grad()
            loss = criterion(encoded,seq[:,seq_index])
            loss.backward()

            clip_grad_norm(encoder.parameters(),max_norm=1)

            #decoder_optimizer.step()
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()

            class_prediction = torch.argmax(encoded,dim=1)

            print('------------------------------new step---------------------------------')    
            print('Loss {}'.format(loss))
            print('Probability prediction {}'.format(encoded))
            print('Class prediction {}'.format(class_prediction))
            print('Ground truth {}'.format(seq[:,seq_index]))

            
            




        # [1, 512 , 4, 4, 4]
       
        


for epoch in range(0,NUM_EPOCHS):

    train(dataloader = dataloader,
        encoder = encoder,
        decoder = decoder,
        criterion = CRITERION,
        encoder_optimizer = encoder_optimizer,
        decoder_optimizer = None,
        epoch=epoch)


