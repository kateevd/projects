
# coding: utf-8

# In[1]:


import os
import sys
import glob
import time
import math
import tqdm
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import extractors


# # PSPNet architecture

# In[3]:


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        
        return self.final(p)


# In[4]:


net = PSPNet(sizes=(1, 2, 3, 6), n_classes=1, psp_size=512, deep_features_size=256, backend='resnet18')
net.cuda()
print('net compiled')


# # Unet Architecture

# In[5]:


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x


# In[6]:


#net = UNet(3, 1)
#net.cuda()
#print('net compiled')


# # Data loader

# In[7]:


def preprocess(img):
    return (img / 255.) * 2. - 1.

def fake_preprocess(img):
    return img / 255.

def res_preprocess(img):
    img = img / 255.
    img[:,:,0] = (img[:,:,0] - 0.406) / 0.225
    img[:,:,1] = (img[:,:,1] - 0.456) / 0.224
    img[:,:,2] = (img[:,:,2] - 0.485) / 0.229
    
    return img

def crop_kek(merged):
    kek_size = 512
    h = np.random.randint(0, 2048 - kek_size)
    w = np.random.randint(0, 2560 - kek_size)
    return merged[h:h+kek_size,w:w+kek_size,:]


# In[8]:


class CircuitsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.masks = [os.path.basename(x) for x in glob.glob(root_dir + 'masks/*.npy')]
        self.imgs = [x.split('.npy')[0] + '.png' for x in self.masks]
        
        del_img = []
        del_msk = []
        for image, mask in zip(self.imgs, self.masks):
            if not os.path.exists(self.root_dir + 'imgs/' + image):
                del_img.append(image)
                del_msk.append(mask)
        for i in range(len(del_img)):
            self.masks.remove(del_msk[i])
            self.imgs.remove(del_img[i])

        self.transform = transform
        
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        img = cv2.imread(root_dir + 'imgs/' + self.imgs[idx]).astype(np.float32)
        img = res_preprocess(img)[:,:,::-1]
        
        with open(root_dir + 'masks/' + self.masks[idx], 'rb') as fp:
            msk = np.load(fp)
        
        combined = np.concatenate([img, msk[:,:,np.newaxis]], axis=2)
        combined = cv2.resize(combined, (512, 512))
        
        if self.transform:
            combined = self.transform.augment_image(combined)

        combined = combined.transpose(2,0,1)
        return combined[:3,:,:], np.expand_dims(combined[3,:,:], axis=0)


# In[9]:


import imgaug as ia
from imgaug import augmenters as iaa


# In[10]:


sometimes = lambda aug: iaa.Sometimes(0.6, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.Affine(
            scale=(0.6, 1.7),
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
    ))
])


# In[11]:


root_dir = '/mnt/data/datasets/circuits/'


# In[12]:


train_set = CircuitsDataset(root_dir, transform=seq)
train_load = DataLoader(train_set, batch_size=6, shuffle=True, num_workers=1)


# In[13]:


frames, targets = train_set[1]
print(frames.shape)
print(targets.shape)
plt.imshow(frames.transpose(1, 2, 0))
plt.show()
plt.imshow(targets[0,:,:])
plt.show()


# # Training

# In[14]:


def dice_loss(y_pred, y_true):
    smooth = 1.
    intersection = y_pred * y_true
    int_batch_chnl = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
    pred_batch_chnl = torch.sum(torch.sum(y_pred, dim=-1), dim=-1)
    true_batch_chnl = torch.sum(torch.sum(y_true, dim=-1), dim=-1)

    dice_batch_chnl = (2. * int_batch_chnl + smooth) / (pred_batch_chnl + true_batch_chnl + smooth)

    return -torch.mean(dice_batch_chnl)


# In[15]:


optims = optim.Adam(net.parameters())
criter = nn.BCEWithLogitsLoss()


# In[16]:


def train_epoch(train_load):
    net.train()
    sum_loss = 0.
    btchs = 0
    for frames, targets in train_load:
        framevar, targvar = frames.cuda(), targets.cuda()
        optims.zero_grad()
        preds = net(framevar)
        loss = criter(preds, targvar) + dice_loss(F.sigmoid(preds), targvar)
        #loss = dice_loss(F.sigmoid(preds), targvar)
        loss.backward()
        optims.step()
        sum_loss += loss.item()
        btchs += 1
    return sum_loss / btchs

def test_epoch(test_load):
    net.eval()
    sum_loss = 0.
    btchs = 0
    for frames, targets in test_load:
        framevar, targvar = frames.cuda(), targets.cuda()
        preds = net(framevar)
        loss = criter(preds, targvar)# + dice_loss(F.sigmoid(preds), targvar)
        sum_loss += loss.data[0]
        btchs += 1
    return sum_loss / btchs


# In[17]:


def train_net(n_epochs):
    dirname = '/home/mio/Documents/circuits/snaps'
    
    train_history = []
    test_history  = []
    for i in tqdm.tqdm(range(n_epochs)):
        train_history.append(train_epoch(train_load))
        print('Epoch: {} Loss: {}'.format(i, train_history[-1]))
        #print('testing...')
        #test_history.append(test_epoch(test_load))
        #print('Epoch: {} Loss: {} Test: {}'.format(i, train_history[-1], test_history[-1]))
        torch.save(net.state_dict(), dirname + '/circuits_{}_{}.pth'.format(i, train_history[-1]))
    return train_history#, test_history


# In[18]:


tr_loss = train_net(200)


# In[ ]:


net.eval()
for frames, targets in train_load:
    framevar, targvar = frames.cuda(), targets.cuda()
    result = torch.sigmoid(net(framevar))
    heatmaps = result.detach().cpu().numpy()[0]
    plt.imshow(heatmaps[0,:,:])
    plt.show()
    targmaps = targvar.detach().cpu().numpy()[0]
    plt.imshow(targmaps[0,:,:])
    plt.show()


# In[19]:


plt.plot(tr_loss)
plt.show()


# ## Bake masks

# In[ ]:


msks = [os.path.basename(x) for x in glob.glob(root_dir + 'masks/*.png')]
imgs = [os.path.basename(x).split('_mrk') for x in msks]
imgs = [x[0] + x[1] for x in imgs]


# In[ ]:


def get_mask(img):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i,j,0] == 0 and img[i,j,1] == 255 and img[i,j,2] == 0):
                mask[i,j] = 1.0
    return mask


# In[10]:


for mask in msks:
    bin_name = root_dir + 'masks/' + mask.split('_mrk.png')[0] + '.npy'
    with open(bin_name, 'wb') as fp:
        np.save(fp, get_mask(cv2.imread(root_dir + 'masks/' + mask)))

