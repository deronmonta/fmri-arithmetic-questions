from utils import *
from dataset import *
from networks.model import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn


#DATA_DIR = '/media/noahyang/Yale/fMRI_Data/fmri'

# Hyperparameters and configurations

DATA_DIR = './data'
BATCH_SIZE = 2
SEQ_CSV = './sequence.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#CRITERION = nn.CrossEntropyLoss().to(DEVICE)
CRITERION = nn.NLLLoss().to(DEVICE)
NUM_EPOCHS = 20
ENCODER_LR = 1e-2  # learning rate for encoder if fine-tuning
DECODER_LR = 4e-2     # learning rate for decoder

ATTENTION_DIM = 512
DECODER_DIM = 512

NUM_TASK = 5 # Number of different types of class
SEQ_LEN = 10 #Length of prediction duration

fmri_dataset = FMRI_Dataset(DATA_DIR,SEQ_CSV,SEQ_LEN)
dataloader = DataLoader(fmri_dataset, batch_size=1)


cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoder = ResNet34(seq_len=SEQ_LEN) # Dimension ares [batchsize ,num_filter, H, W, Z]
decoder = Decoder(ATTENTION_DIM,DECODER_DIM,seq_len=SEQ_LEN)
decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=DECODER_LR)
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),lr=ENCODER_LR) 



encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)
print(encoder)
print(decoder)


def train(dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    

    encoder.train()
    decoder.train()

    for index, sample in enumerate(dataloader):

        volume = sample['volumes']
        seq = sample['sequence']

        # seq = class2onehot(seq,SEQ_LEN,BATCH_SIZE,NUM_TASK)
        # print('One hot shape {}'.format(seq.shape))



        volume, seq = volume.to(DEVICE), seq.to(DEVICE)
        volume,seq = volume.float(), seq.long()
        

        print('Volume shape {}'.format(volume.shape)) #[batchsize, seqlen, 64, 64, 64]
        print('Sequence {}'.format(seq.shape)) #[batchsize, seqlen]

        encoded = encoder(volume)
        print('Encoded shape: {}'.format(encoded.shape))
        scores, length, alpha = decoder(encoded,decode_length=SEQ_LEN)
        

        scores = scores.float()
        print(scores)
        scores = scores.permute((0,2,1)) 
        

        
        loss = criterion(scores,seq)
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        decoder_optimizer.step()
        encoder_optimizer.step()

        class_prediction = torch.argmax(scores,dim=1)

        print('Loss {}'.format(loss))
        print('Probability prediction {}'.format(scores))
        print('Class prediction {}'.format(class_prediction))
        print('Ground truth {}'.format(seq))

for epoch in range(0,NUM_EPOCHS):

    train(dataloader = dataloader,
        encoder = encoder,
        decoder = decoder,
        criterion = CRITERION,
        encoder_optimizer = encoder_optimizer,
        decoder_optimizer = decoder_optimizer,
        epoch=epoch)


