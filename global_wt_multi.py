gpu=0
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
import pandas as pd
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_allocated())
gpu_id = gpu

# %%
# !nvidia-smi

# %%
torch.cuda.empty_cache()

# %%
device=torch.device("cuda:0")
torch.cuda.set_device(gpu)
print("Training on GPU... Ready for HyperJump...")

# %%
# for gold models
def densenet_Model(pretrained=True):
    model = models.densenet121(pretrained=pretrained) # Returns Defined Densenet model with weights trained on ImageNet
    num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
    model.classifier = nn.Linear(num_ftrs, 1) # Overwrites the Classifier layer with custom defined layer for transfer learning
    model = model.to(device) # Transfer the Model to GPU if available
    return model




# %%
# path="/DATA/chowdari1/saved_models/nih_gold(100epoch).pth"
path="/DATA/chowdari1/saved_models/nih_ppr_gold(100epoch).pth"
model_nih = densenet_Model(pretrained=True)
model_nih=model_nih.to(device)
x_nih=torch.load(path,map_location='cpu')
print('nih: ',path)

# %%
path="/DATA/chowdari1/saved_models/stanford_gold_wow(100epoch).pth"
model_stanford = densenet_Model(pretrained=True)
x_stanford=torch.load(path,map_location='cpu')
print('stanford: ',path)

path="/DATA/chowdari1/saved_models/vinbig_15k(100epoch).pth"
model_vb = densenet_Model(pretrained=True)
x_vb=torch.load(path,map_location='cpu')
print('vinbig: ',path)


# %%


# %%


# %%
# print("s_stanford: ")
# print(x_stanford[0])

# %%
# print("x_nih: ")
# print(x_nih)


model_global = densenet_Model(pretrained=True)
x_global=torch.load(path)

keys=x_stanford.keys()
len(keys)

simpleavg={'path': '/DATA/chowdari1/saved_models/global/glb_3_wow_simpleavg.pth',
        'p':1, 'q':1 ,'r':1}
trainsize={'path': '/DATA/chowdari1/saved_models/global/glb_3_wow_trainsize.pth',
        'p':8574, 'q':150000 ,'r':7000}
trainloss={'path': '/DATA/chowdari1/saved_models/global/glb_3_wow_trainloss.pth',
        'p': 1/0.1521, 'q':1/0.1515 ,'r':1/0.0406}
valloss={'path': '/DATA/chowdari1/saved_models/global/glb_3_wow_valloss.pth',
        'p': 1/0.3221, 'q':1/0.1660 ,'r':1/0.1235}

mana=[simpleavg, trainsize, trainloss, valloss]

count=0
for dict in mana:
    print('\ncount: ',count)
    count+=1
    print('---'*10)
    path=dict['path']
    p=dict['p']
    q=dict['q']
    r=dict['r']
    sum=p+q+r
    abc=0
    print('start')
    x_nih=torch.load("/DATA/chowdari1/saved_models/nih_ppr_gold(100epoch).pth",map_location='cpu')
    x_stanford=torch.load("/DATA/chowdari1/saved_models/stanford_gold_wow(100epoch).pth",map_location='cpu')
    x_vb=torch.load("/DATA/chowdari1/saved_models/vinbig_gold_21k(100epoch).pth",map_location='cpu')

    for i in keys:
        x_nih[i]=x_nih[i].to(device)
        a=x_nih[i].float()
        if(abc==0):
            print('nih: ',a[0][0][0])
        a*=p/sum

        x_stanford[i]=x_stanford[i].to(device)
        b=x_stanford[i].float()
        if(abc==0):
            print('stanford: ',b[0][0][0])
        b*=q/sum

        x_vb[i]=x_vb[i].to(device)
        c=x_vb[i].float()
        if(abc==0):
            print('vinbig: ',c[0][0][0])
        c*=r/sum

        x_global[i]=torch.add(a,b)
        x_global[i]=torch.add(x_global[i],c)
        x_global[i]=x_global[i].float()
        if(abc==0):
            print('global: ',x_global[i][0][0][0])
            abc=1
    
    # # %%
    # path="/DATA/chowdari1/saved_models/ppr/avg_global_ppr.pth"
    model_global = densenet_Model(pretrained=True)
    model_global.load_state_dict(x_global)
    torch.save(model_global.state_dict(), path)
    print('weights: ,nih:',p/sum,' stanf: ',q/sum,' vinbig:',r/sum)
    print('model saved as: ',path)