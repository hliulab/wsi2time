# !/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import stats
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.feature_dataset import FeatureDataset_mo
from models.att_model import CLAM_SB_TIME_NN_Pool
from utils.early_stopping_utils import EarlyStopping

#python train_reg_att.py --exp moco_init_r --epochs 30 --lr 0.001 --csv_path precess_data/breast_fpkm_ts_log10+1.brca_csv
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--feature_path', default=r'../features', type=str,
                    help='path to feature')
parser.add_argument('--exp',type=str,default='None')
parser.add_argument('--save_path',type=str,default='../result')
parser.add_argument('--csv_path',type=str,default='../dataset_csv/sample_data.csv')
parser.add_argument('--epcohs',type=int,default=200)
parser.add_argument('--lr',type=float,default=0.003)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--device',type=str,default='cuda')
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
train_set = FeatureDataset_mo(feature_path=args.feature_path,
                           data_path=args.csv_path,
                           dataset_type='train', is_normalized=False)
valid_set = FeatureDataset_mo(feature_path=args.feature_path,
                           data_path=args.csv_path,
                           dataset_type='valid', is_normalized=False)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

model = CLAM_SB_TIME_NN_Pool(n_classes=21)
epochs = args.epcohs

def train(num_epochs: int = epochs, checkpoint_path: str = None) -> None:
    model.to(device)
   # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loss_f1 = nn.L1Loss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        del checkpoint['classifiers.weight'], checkpoint['classifiers.bias']
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = 1
        print('有保存模型！')
    else:
        start_epoch = 1
        print('无保存模型，将从头开始训练！')

    result_train_loss = []
    result_valid_loss = []
    
    savepath = os.path.join(args.save_path, args.exp)
    os.makedirs(savepath,exist_ok=True)
    early_stopping = EarlyStopping(patience=20, stop_epoch=100, verbose=False)
    print("Begin training...")
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        running_train_L1_loss = 0.0
        running_val_L1_loss = 0.0
        
        # Training Loop
        train_label = []
        train_predicted = []
        for iteration,data in enumerate(tqdm(train_loader, desc="train...")):
            # for data in enumerate(train_loader, 0):
            wsi_feature, label_data = data
            optimizer.zero_grad()  # zero the parameter gradients
            wsi_feature = torch.squeeze(wsi_feature)
           
            wsi_feature = wsi_feature.to(device)
            predicted_outputs = model(wsi_feature)  # predict output from the model
            predicted_outputs = predicted_outputs.to(device)
            label_data = label_data.to(device)
            # protein_data = protein_data.squeeze()
            train_loss = loss_fn(predicted_outputs, label_data)  # calculate loss for the predicted output
            L1_loss = loss_f1(predicted_outputs, label_data)
            
            train_loss.backward()  # back propagate the loss
            optimizer.step()
            
            running_train_loss += train_loss.item()  # track the loss value
            running_train_L1_loss += L1_loss.item()
            train_label.append(label_data.cpu().detach().numpy())
            train_predicted.append(predicted_outputs.cpu().detach().numpy())
        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)
        running_train_L1_loss = running_train_L1_loss / len(train_loader)
       
        # for i in range(21):
        #     corr, p = stats.pearsonr(np.array(train_predicted).squeeze()[:,i], np.array(train_label).squeeze()[:,i])
        #     print(corr, p)

        scheduler.step()

        # Validation Loop
        with torch.no_grad():
            model.eval()
            train_label = []
            train_predicted = []
            for iteration,data in enumerate(tqdm(val_loader, desc="val...")):
                wsi_feature, label_data = data
                wsi_feature = torch.squeeze(wsi_feature)
                wsi_feature = wsi_feature.to(device)
                predicted_outputs = model(wsi_feature)
                label_data = label_data.to(device)
                # protein_data = protein_data.squeeze()
                predicted_outputs = predicted_outputs.to(device)
                val_loss = loss_fn(predicted_outputs, label_data)
                L1_loss = loss_f1(predicted_outputs, label_data)
                # The label with the highest value will be our prediction
                running_val_L1_loss += L1_loss.item()
                running_val_loss += val_loss.item()
                train_label.append(label_data.cpu().detach().numpy())
                train_predicted.append(predicted_outputs.cpu().detach().numpy())
        # Calculate validation loss value
        val_loss_value = running_val_loss / len(val_loader)
        running_val_L1_loss = running_val_L1_loss / len(val_loader)
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total_val
        # number of predictions done.
        # Save the model if the accuracy is the best

        # Print the statistics of the epoch
        print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value,
              'Train L1Loss is : %.4f' % running_train_L1_loss,
              'Validation Loss is: %.4f' % val_loss_value, 'Val L1Loss is : %.4f' % running_val_L1_loss)

        result_train_loss.append(train_loss_value)
        result_valid_loss.append(val_loss_value)

        early_stopping(epoch, val_loss_value, model, optimizer,
                       ckpt_name=os.path.join(savepath,'checkpoint_brca_ts.pth.tar'))
        if early_stopping.early_stop:
            print("该停了")
            break

    plt.plot(range(1, len(result_train_loss) + 1), result_train_loss, label='train loss', color='r')
    plt.plot(range(1, len(result_valid_loss) + 1), result_valid_loss, label='valid loss', color='g')
    plt.legend()
    plt.savefig(os.path.join(savepath,'att_ReLU_brca_ts.png'))


if __name__ == "__main__":
    # a = torch.rand((1030, 1024))
    # z = model(a)
    # print(z.shape)
    train()
