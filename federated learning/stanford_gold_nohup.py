# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
gpu=6
CUDA_VISIBLE_DEVICES=gpu
# get_ipython().system('nvidia-smi')


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


# %%
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
df=pd.read_csv('/DATA/chowdari1/DATA/csv/mana_great_all_stanford.csv')




# %%
df.shape

# %%
df["target"].value_counts()


# %%
df=df.sample(frac=1)

print('train--')
print(df.iloc[:28500]['target'].value_counts())
print('val--')
print(df.iloc[28500:57000]['target'].value_counts())
print('test--')
print(df.iloc[57000:]['target'].value_counts())


# %%



# %%
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
normalizer=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
transformations = {
		'train': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.25, contrast=0.25),
			transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
			transforms.ToTensor(),
			transforms.Normalize(normalizer[0], normalizer[1])]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(normalizer[0], normalizer[1])]),
		'test': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(normalizer[0], normalizer[1])])	
			}


# %%

train_data=CustomImageDataset(df,'/DATA/chowdari1/DATA/dataset/stanford/',transformations["train"],0,28500)
valid_data=CustomImageDataset(df,'/DATA/chowdari1/DATA/dataset/stanford/',transformations["val"],28500,57000)
test_data=CustomImageDataset(df,'/DATA/chowdari1/DATA/dataset/stanford/',transformations["test"],57000,64088)
print(len(train_data))
print(len(test_data))
trainloader=torch.utils.data.DataLoader(train_data, batch_size=64)
valloader=torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader=torch.utils.data.DataLoader(test_data, batch_size=64)
print(trainloader)
print(len(testloader))
dataloaders = {"train":trainloader, "val":valloader, "test": testloader}

data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val','test']}
data_sizes


# %%
import torch
torch.cuda.empty_cache()
device=torch.device("cuda:6")
print("Training on GPU... Ready for HyperJump...")


# %%
torch.cuda.set_device(gpu)

# %%
def densenet_Model(pretrained=True):
    model = models.densenet121(pretrained=pretrained) # Returns Defined Densenet model with weights trained on ImageNet
    num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
    model.classifier = nn.Linear(num_ftrs, 1) # Overwrites the Classifier layer with custom defined layer for transfer learning
    model = model.to(device) # Transfer the Model to GPU if available
    return model

model = densenet_Model(pretrained=True)

# specify loss function (categorical cross-entropy loss)
criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda() 

# Specify optimizer which performs Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0001, momentum=0.9)		

# Learning Scheduler
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience = 5)

# %%
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))


import datetime

# %%
from sklearn.metrics import roc_auc_score
def epoch_train(model,optimizer, criterion):
    model.train()
    loss_train = 0
    loss_train_norm = 0
    loss_tensor_mean_train = 0
    output_list = []
    label_list = []
    phase="train"
    for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        # labels = labels.type(torch.FloatTensor) 
        labels = labels.to(device, non_blocking=True)
        labels = labels.view(labels.size()[0],-1)

        optimizer.zero_grad()			
        outputs = model(inputs)
        # _, preds = torch.max(outputs.data, 1)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            score = torch.sigmoid(outputs)
        else:
            score = torch.sigmoid(outputs)
        preds = score>0.5
        preds = preds.type(torch.cuda.LongTensor)
        
        labels = labels.type(torch.cuda.FloatTensor) #add for BCE loss
        loss = criterion(outputs, labels)
        loss_tensor_mean_train += loss

        labels = labels.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()

        for i in range(outputs.shape[0]):
            output_list.append(outputs[i].tolist())
            label_list.append(labels[i].tolist())

        loss_train_norm += 1
        loss.backward()
        optimizer.step()
    loss_tensor_mean_train = np.float(loss_tensor_mean_train) / loss_train_norm
    epoch_auc =  roc_auc_score(np.array(label_list), np.array(output_list))
    output_list = []
    label_list = []
    return loss_tensor_mean_train, epoch_auc


# %%
def epoch_val(model, criterion):
    model.eval()
    loss_val = 0
    loss_val_norm = 0
    loss_tensor_mean_val = 0
    output_list = []
    label_list = []
    phase='val'
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase, leave=False):
            labels = labels.type(torch.FloatTensor) #add for BCE loss
            inputs = inputs.cuda(gpu_id, non_blocking=True)
            labels = labels.cuda(gpu_id, non_blocking=True)
            labels = labels.view(labels.size()[0],-1) #add for BCE loss

            outputs = model(inputs)
            loss_tensor = criterion(outputs, labels)
            loss_tensor_mean_val += loss_tensor
            
            labels = labels.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            for i in range(outputs.shape[0]):
                output_list.append(outputs[i].tolist())
                label_list.append(labels[i].tolist())
            loss_val_norm += 1
    loss_tensor_mean_val = np.float(loss_tensor_mean_val) / loss_val_norm
    epoch_auc =  roc_auc_score(np.array(label_list), np.array(output_list))
    output_list = []
    label_list = []
    return loss_tensor_mean_val, epoch_auc

    


# %%
def train_model(model, criterion, optimizer, scheduler, num_epoch=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_min = np.inf

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch+1, num_epoch))
        print(datetime.datetime.now())
        print('-' * 10)


        loss_train, auc_train = epoch_train(model, optimizer, criterion)
        loss_val, auc_val=epoch_val(model,criterion)

        scheduler.step(loss_val)
        if loss_val < loss_min:
            print('Val loss Decreased from {:.4f} to {:.4f} \nSaving Weights... '.format(loss_min, loss_val))
            loss_min=loss_val
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

        print ('*'*20)	
        print ('Train_AUC: {:.4f}     Train_loss: {:.4f}'            .format(auc_train, loss_train))
        print ('  Val_AUC: {:.4f}     Val_loss: {:.4f}'            .format(auc_val, loss_val))
        print ('\n')
    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val loss: {:.4f}'.format(loss_min))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model
    
        
           


# %%
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %%
train_df=pd.DataFrame()
test_df=pd.DataFrame()
testloader=[]


# %%
torch.cuda.empty_cache()


# %%
base_model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epoch=100)


# %%
path="/DATA/chowdari1/saved_models/stanford_gold(100epoch).pth"
torch.save(base_model.state_dict(), path)

print('model is saved as: stanford_gold(100epoch).pth ')
