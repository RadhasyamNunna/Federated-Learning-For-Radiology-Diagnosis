# %%
gpu=7
CUDA_VISIBLE_DEVICES=gpu

# %%
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import time
import copy
from random import shuffle

import tqdm.notebook as tqdm

import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report
from PIL import Image
import cv2

# import osa
import shutil
import seaborn as sns
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from glob import glob
from sklearn.metrics import roc_curve,auc, precision_score,precision_recall_curve,recall_score,precision_recall_fscore_support,confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models
from prettytable import PrettyTable
print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore")
# import pandas as pd
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_allocated())
gpu_id = gpu

# %%
train_df=pd.read_csv("/DATA/chowdari1/DATA/csv/ppr_nih_train.csv")
val_df=pd.read_csv("/DATA/chowdari1/DATA/csv/ppr_nih_val.csv")
test_df=pd.read_csv("/DATA/chowdari1/DATA/csv/ppr_nih_test.csv")
train_df=train_df.sample(frac=1,random_state=172)
val_df=val_df.sample(frac=1,random_state=172)
test_df=test_df.sample(frac=1,random_state=172)

# %%
# val_df.head()

# %%
# val_df.head()

# %%
# train_df=train_df.sample(frac=1,random_state=172)
# val_df=val_df.sample(frac=1,random_state=172)

# %%
train_df.head()

# %%

# for i in range(df.shape[0]):
#     df['path'][i]+='.png'
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform,start,count):
        self.ipaths = df["path"][start:count].to_numpy()
        self.target=df["target"][start:count].to_numpy()
        self.transform = transform
        self.count=count
        self.img_dir=img_dir


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img_path=self.img_dir+self.ipaths[idx]
        image = Image.open(img_path).convert('RGB')
        label=self.target[idx]
        image = self.transform(image)
        return image, label

# %%
# df.iloc[0]

# %%
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.449,),(0.226,))
])

# %%
train_df.shape, val_df.shape

# %%

train_data=CustomImageDataset(train_df,'',transform,0,8448)
valid_data=CustomImageDataset(val_df,'',transform,0,1536)
# test_data=CustomImageDataset(df,'/DATA/dataset/vinbig/vinbig/trainpng/',transform,7000+6000,df.shape[0])
print(len(train_data))
# print(len(test_data))
trainloader=torch.utils.data.DataLoader(train_data, batch_size=256)
valloader=torch.utils.data.DataLoader(valid_data, batch_size=256)
# testloader=torch.utils.data.DataLoader(test_data, batch_size=256)
print(trainloader)
# print(len(testloader))
# dataloaders = {"train":trainloader, "val":valloader, "test": testloader}
dataloaders = {"train":trainloader, "val":valloader}
# data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val','test']}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val']}

print(data_sizes)

# %%

# %%


# %%
import torch
torch.cuda.empty_cache()
device=torch.device("cuda:7")
print("Training on GPU... Ready for HyperJump...")

# %%
torch.cuda.set_device(gpu_id)

# %%
for img,label in trainloader:
  print(img.shape)
  break
for img,label in valloader:
  print(img.shape)
  break

# %%
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        # out_width = (28+2-5)/2+1 = 27/2+1 = 13
        self.conv2 = nn.Conv2d(8,16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        # out_width = (14-5)/2+1 = 5
        #self.drop1=nn.Dropout2d(p=0.3) 
        # 6 * 6 * 16 = 576
        self.linear1 = nn.Linear(23328, 224)
        self.linear2 = nn.Linear(224, latent_dims)
        self.linear3 = nn.Linear(224, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        #print(x.shape)
        global en
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # print(en,'hello',z[0])
        # en+=1
        return z


# %%
class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(latent_dims, 224),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(224, 23328),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 27, 27))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        global de
        # Apply linear layers
        x = self.decoder_lin(x)
        # print('de1',x)
        # Unflatten
        x = self.unflatten(x)
        # print('de2',x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # print('de3',x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        # print(de,'de4',x[0][0][0][0])
        # de+=1
        return x


# %%
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        # print(z)
        return self.decoder(z)

# %%
torch.manual_seed(0)

d = 4

vae = VariationalAutoencoder(latent_dims=d)

lr = 1e-3 

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)

# %%
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    phase='train'
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    # for x, _ in dataloader: 
    for x, _ in tqdm.tqdm(dataloader, desc=phase, leave=False):
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

# %%
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    phase='test'
    with torch.no_grad(): # No need to track the gradients
        for x, _ in tqdm.tqdm(dataloader, desc=phase, leave=False):
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


# %%
# num_epochs = 10
import datetime
def train(vae,num_epochs):
    best_test_loss = float('inf')
    best_model_wts = copy.deepcopy(vae.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format( epoch+1, num_epochs))       
        print (datetime. datetime. now())
        print('-' * 10)
        train_loss = train_epoch(vae,device,trainloader,optim)
        val_loss = test_epoch(vae,device,valloader)
        if(best_test_loss > val_loss):
            best_test_loss=val_loss
            best_model_wts = copy.deepcopy(vae.state_dict())
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    print('best_test_loss: ',best_test_loss)
    # plot_ae_outputs(vae.encoder,vae.decoder,n=5)

    vae.load_state_dict(best_model_wts)
    return vae

# %%
va=train(vae,100)

# %%
path="/DATA/chowdari1/saved_models/vae/vae_big_nihppr_100ep.pth"
torch.save(vae.state_dict(), path)
print(path)

# %%
test_data=CustomImageDataset(test_df,'',transform,0,1280)
testloader=torch.utils.data.DataLoader(test_data, batch_size=256)
len(testloader.sampler)
# test_loss = test_epoch(model,device,testloader)
# test_loss

# %%
test_loss = test_epoch(va,device,testloader)
print('test_loss: ',test_loss)

# %%
from torchvision.utils import save_image
def plot_ae_outputs(encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_data[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      name='./images/nihppr_100_'+str(i)+'in.jpeg'      
      save_image(img, name)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray') 
      name='./images/nihppr_100_'+str(i)+'out.jpeg'      
      save_image(rec_img, name) 
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show() 

plot_ae_outputs(vae.encoder,vae.decoder,n=5)
print('images added as: ./images/nihppr_100_')



