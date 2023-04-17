from __future__ import print_function, division 
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import pickle
import pandas as pd
from PIL import Image
import argparse
from apex import amp
from sklearn.metrics.ranking import roc_auc_score
from models.modeling_irene import IRENE, CONFIGS
from tqdm import tqdm
import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

tk_lim = 40

disease_list = ['COPD', 'Bronchiectasis', 'Pneumothorax', 'Pneumonia', 'ILD', 'Tuberculosis', 'Lung cancer', 'Pleural effusion']

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading IRENE...")
    return model

def computeAUROC (dataGT, dataPRED, classCount=8):
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, k) + '.png'
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        cc = torch.from_numpy(self.mm_data[k]['pdesc']).float()
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        lab = torch.from_numpy(self.mm_data[k]['bts']).float()
        return img, label, cc, demo, lab

def test(args):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["IRENE"]
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes)
    irene = load_weights(model, 'model.pth')
    img_dir = args.DATA_DIR

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    test_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    optimizer_irene = torch.optim.AdamW(irene.parameters(), lr=3e-5, weight_decay=0.01)
    irene, optimizer_irene = amp.initialize(irene.cuda(), optimizer_irene, opt_level="O1")

    irene = torch.nn.DataParallel(irene)

    #----- Test ------
    print('--------Start testing-------')
    irene.eval()
    with torch.no_grad():
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, cc, demo, lab = data
            cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = irene(imgs, cc, lab, sex, age)[0]

            probs = torch.sigmoid(preds)
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()
        
        print('mean AUROC:' + str(aurocMean))
         
        for i in range (0, len(aurocIndividual)):
            print(disease_list[i] + ': '+str(aurocIndividual[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    args = parser.parse_args()
    test(args)
