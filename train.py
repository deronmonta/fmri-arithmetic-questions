from utils import *
from dataset import *
from networks.model import *
#from networks.blocks import *
#DATA_DIR = '/media/noahyang/Yale/fMRI_Data/fmri'
DATA_DIR = './data'
BATCH_SIZE = 2

fmri_dataset = FMRI_Dataset(DATA_DIR)
dataloader = DataLoader(fmri_dataset, batch_size=1)

encoder = Encoder(num_in_channel=1,num_filter=8)
#encoder = resnet10(sample_size=64,sample_duration=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

encoder = encoder.to(device)
print(encoder)

for index, sample in enumerate(dataloader):

    #print(sample)

    volume = sample['volumes'][0]

    # Need to unsqueeze twice, once for batch size one for channel
    volume = volume.unsqueeze(0)
    volume = volume.unsqueeze(0)

    print(volume.shape)

    output = encoder(volume)
    print('Output shape {}'.format(output.shape))
