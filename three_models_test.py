gpu=5
CUDA_VISIBLE_DEVICES=gpu
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
import pandas as pd
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_allocated())
gpu_id = gpu

torch.cuda.empty_cache()
device=torch.device("cuda:3")
torch.cuda.set_device(gpu)
print("Training on GPU... Ready for HyperJump...")

def densenet_Model(pretrained=True):
    model = models.densenet121(pretrained=pretrained) # Returns Defined Densenet model with weights trained on ImageNet
    num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
    model.classifier = nn.Linear(num_ftrs, 1) # Overwrites the Classifier layer with custom defined layer for transfer learning
    model = model.to(device) # Transfer the Model to GPU if available
    return model


model_nih = densenet_Model(pretrained=True)
model_nih.load_state_dict(torch.load("/DATA/chowdari1/saved_models/nih_gold(100epoch).pth"))
model_nih.eval()

model_stanford = densenet_Model(pretrained=True)
model_stanford.load_state_dict(torch.load("/DATA/chowdari1/saved_models/stanford_gold(100epoch).pth"))
model_stanford.eval()

model_vinbig = densenet_Model(pretrained=True)
model_vinbig.load_state_dict(torch.load("/DATA/chowdari1/saved_models/vinbig_15k(100epoch).pth"))
model_vinbig.eval()

model_global = densenet_Model(pretrained=True)
dhaari='/DATA/chowdari1/saved_models/ppr/three_train_loss_avg.pth'
model_global.load_state_dict(torch.load(dhaari))
model_global.eval()
print(dhaari)


normalizer=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
transformations=torchvision.transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(normalizer[0], normalizer[1])])

from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, df,img_dir, transform,start,count):
        self.ipaths = df["path"][start:count].to_numpy()
        self.target=df["target"][start:count].to_numpy()
        self.img_dir = img_dir
        self.transform = transform
        self.count=count


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img_path=self.img_dir+self.ipaths[idx]
        image = Image.open(img_path).convert('RGB')
        label=self.target[idx]
        image = self.transform(image)
        return image, label


#NIH
df_nih =pd.read_csv('/DATA/chowdari1/DATA/csv/nih_test.csv')
print(df_nih["target"].value_counts())

test_data=CustomImageDataset(df_nih,"",transformations,0,25596)

print(len(test_data))
testloader=torch.utils.data.DataLoader(test_data, batch_size=64)
print(len(testloader))

import datetime
now = datetime. datetime. now()
from sklearn import metrics
def test(model,dataloader):
    print (datetime. datetime. now())
    model.eval()
    running_corrects = 0
    output_list =[]
    label_list = []
    preds_list = []
    phase='test'
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader, leave=False):
            labels_auc = labels
            labels_print = labels
            labels_auc = labels_auc.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor) #add for BCE loss
            inputs = inputs.cuda(gpu_id, non_blocking=True)
            labels = labels.cuda(gpu_id, non_blocking=True)
            labels_auc = labels_auc.cuda(gpu_id, non_blocking=True)

            labels = labels.view(labels.size()[0],-1) #add for BCE loss
            labels_auc = labels_auc.view(labels_auc.size()[0],-1) #add for BCE loss
            # forward
            outputs = model(inputs)
            # _, preds = torch.max(outputs.data, 1)
            score = torch.sigmoid(outputs)
            score_np = score.data.cpu().numpy()
            preds = score>0.5
            preds_np = preds.data.cpu().numpy()
            preds = preds.type(torch.cuda.LongTensor)

            labels_auc = labels_auc.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()
            for i in range(outputs.shape[0]):
                output_list.append(outputs[i].tolist())
                label_list.append(labels_auc[i].tolist())
                preds_list.append(preds_np[i].tolist())
            # running_corrects += torch.sum(preds == labels.data)
            # labels = labels.type(torch.cuda.FloatTensor)
            running_corrects += torch.sum(preds.data == labels.data) #add for BCE loss
    data_size=len(dataloader.sampler)
    acc = np.float(running_corrects) / data_size
    auc = metrics.roc_auc_score(np.array(label_list), np.array(output_list), average=None)
    # print(auc)
    fpr, tpr, _ = metrics.roc_curve(np.array(label_list), np.array(output_list))
    roc_auc = metrics.auc(fpr, tpr)

    ap = metrics.average_precision_score(np.array(label_list), np.array(output_list))

    tn, fp, fn, tp = metrics.confusion_matrix(label_list, preds_list).ravel()

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    sensitivity = recall
    specificity = tn/(tn+fp)
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)

    print(classification_report(label_list,preds_list))
    print('Test Accuracy: {0:.4f}  Test AUC: {1:.4f}  Test_AP: {2:.4f}'.format(acc, auc, ap))
    print('TP: {0:}  FP: {1:}  TN: {2:}  FN: {3:}'.format(tp, fp, tn, fn))
    print('Sensitivity: {0:.4f}  Specificity: {1:.4f}'.format(sensitivity, specificity))
    print('Precision: {0:.2f}%  Recall: {1:.2f}%  F1: {2:.4f}'.format(precision*100, recall*100, f1))
    print('PPV: {0:.4f}  NPV: {1:.4f}'.format(PPV, NPV))
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of abnormal/normal classification: ')
    plt.legend(loc="lower right")
    # plt.savefig('ROC_abnormal_normal_cls_'+args.arch+'_'+args.test_labels+'.pdf', bbox_inches='tight')
    plt.show()
    return fpr,tpr


print("NIH model on NIH_test")
nihfpr_nih,nihtpr_nih=test(model_nih,testloader)

print("STANFORD model on NIH_test")
nihfpr_st,nihtpr_st=test(model_stanford,testloader)

print("VINBIG model on NIH_test")
nihfpr_vb,nihtpr_vb=test(model_vinbig,testloader)

print("GLOBAL model on NIH_test")
nihfpr_glb,nihtpr_glb=test(model_global,testloader)




print("STANFORD")
df_stanford =pd.read_csv('/DATA/chowdari1/DATA/csv/mana_great_all_stanford.csv')
print(df_stanford["target"].value_counts())

test_data_stanford=CustomImageDataset(df_stanford,'/DATA/chowdari1/DATA/dataset/stanford/',transformations,57000,64088)
print(len(test_data_stanford))
testloader_stanford=torch.utils.data.DataLoader(test_data_stanford, batch_size=64)
print(len(testloader_stanford))


print("NIH model on STANFORD_test")
stfpr_nih,sttpr_nih=test(model_nih,testloader_stanford)

print("STANFORD model on STANFORD_test")
stfpr_st,sttpr_st=test(model_stanford,testloader_stanford)

print("VINBIG model on STANFORD_test")
stfpr_vb,sttpr_vb=test(model_vinbig,testloader_stanford)

print("GLOBAL model on STANFORD_test")
stfpr_glb,sttpr_glb=test(model_global,testloader_stanford)






print("VINBIG")
df_vinbig = pd.read_csv("/DATA/dataset/vinbig/vinbig/train_15k.csv")
print(df_vinbig["target"].value_counts())

test_data_vinbig=CustomImageDataset(df_vinbig,'/DATA/dataset/vinbig/vinbig/trainpng/',transformations,13000,df_vinbig.shape[0])
print(len(test_data_vinbig))
testloader_vinbig=torch.utils.data.DataLoader(test_data_vinbig, batch_size=64)
print(len(testloader_vinbig))


print("NIH model on VINBIG_test")
vbfpr_nih,vbtpr_nih=test(model_nih,testloader_vinbig)

print("STANFORD model on VINBIG_test")
vbfpr_st,vbtpr_st=test(model_stanford,testloader_vinbig)

print("VINBIG model on VINBIG_test")
vbfpr_vb,vbtpr_vb=test(model_vinbig,testloader_vinbig)

print("GLOBAL model on VINBIG_test")
vbfpr_glb,vbtpr_glb=test(model_global,testloader_vinbig)