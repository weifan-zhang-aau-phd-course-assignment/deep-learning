#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:12:47 2020

@author: ivl@es.aau.dk
"""

# Assignment: Convolutional neural network.
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt


# For reproducible results.
torch.manual_seed(0)


# ------------------ #
# MODEL ARCHITECTURE #
# ------------------ #
class Convolutional(torch.nn.Module):
    
    def __init__(self, no_channels, kernel_size, pooling_size, flattened_size, hidden_size, output_size):
        super(Convolutional, self).__init__()
        self.no_channels = no_channels
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.flattened_size = flattened_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = torch.nn.Conv2d(1, self.no_channels, self.kernel_size).cuda()
        self.conv2 = torch.nn.Conv2d(self.no_channels, int(self.no_channels/2), self.kernel_size).cuda()
        self.pool = torch.nn.MaxPool2d(self.pooling_size, self.pooling_size).cuda()
        self.fc1 = torch.nn.Linear(self.flattened_size, self.hidden_size).cuda()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size).cuda()
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0),1,self.flattened_size)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# --------------------------- #
# ACCURACY COMPUTATION METHOD #
# --------------------------- #
def compute_accuracy(y_pred, y_labels):
    y_pred = y_pred.cpu().data.numpy()
    y_pred = y_pred.argmax(axis=2)
    acc = y_pred[:,0] - y_labels.cpu().data.numpy()
    return 100*np.sum(acc==0)/len(acc)


# ---------------- #
# DATA PREPARATION #
# ---------------- #
print('Preparing training data...')
# Training data.
X_train = pickle.load(open('X_train.p', 'rb'))
Y_train_oh = pickle.load(open('Y_train.p', 'rb'))
# Reshaping training data.
no_ts = X_train.shape[0] # No. of training examples.
no_tf = X_train.shape[1] # No. of time frames.
no_qb = X_train.shape[2] # No. of quefrency bins.
X_train = np.expand_dims(X_train,axis=1)
# Data normalization.
uX = np.mean(X_train)
sX = np.std(X_train)
X_train = (X_train - uX) / sX
# From one-hot encoding to integers.
Y_train = np.zeros(no_ts)
for i in range(no_ts):
    Y_train[i] = np.where(Y_train_oh[i]==1)[0][0]
# To PyTorch tensors.
X_train = torch.cuda.FloatTensor(X_train)
Y_train = torch.cuda.LongTensor(Y_train)
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_data, batch_size=1024, shuffle=True)
# Validation data.
X_valid = pickle.load(open('X_valid.p', 'rb'))
X_valid = np.expand_dims(X_valid,axis=1)
Y_valid_oh = pickle.load(open('Y_valid.p', 'rb'))
no_vs = X_valid.shape[0] # No. of validation examples.
X_valid = (X_valid - uX) / sX
Y_valid = np.zeros(no_vs)
for i in range(no_vs):
    Y_valid[i] = np.where(Y_valid_oh[i]==1)[0][0]
X_valid = torch.cuda.FloatTensor(X_valid)
Y_valid = torch.cuda.LongTensor(Y_valid)
# Test data.
X_test = pickle.load(open('X_test.p', 'rb'))
X_test = np.expand_dims(X_test,axis=1)
Y_test_oh = pickle.load(open('Y_test.p', 'rb'))
no_es = X_test.shape[0] # No. of test examples.
X_test = (X_test - uX) / sX
Y_test = np.zeros(no_es)
for i in range(no_es):
    Y_test[i] = np.where(Y_test_oh[i]==1)[0][0]
X_test = torch.cuda.FloatTensor(X_test)
Y_test = torch.cuda.FloatTensor(Y_test)


# ---------------- #
# MODEL DEFINITION #
# ---------------- #
print('Defining the model...')
no_cl = 3 # No. of classes.
pooling = 2 # Pooling size.
ksize = 5 # Kernel size.
no_ch = 32 # No. of channels.
# ------------------------------------------
# We do the calculations for flattened_size.
# ------------------------------------------
hs = np.floor((no_tf-ksize+1)/pooling)
hs = np.floor((hs-ksize+1)/pooling)
ws = np.floor((no_qb-ksize+1)/pooling)
ws = np.floor((ws-ksize+1)/pooling)
flattened_size = int(hs*ws*no_ch/2)
# ------------------------------------------
model = Convolutional(no_ch,ksize,pooling,flattened_size,128,no_cl)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters())


# -------------- #
# MODEL TRAINING #
# -------------- #
print('Training the model...')
model.train()
no_epoch = 50 # No. of training epochs.
train_loss = []
val_loss = []
train_acc = []
val_acc = []
for epoch in range(no_epoch):
    # Mini-batch processing.
    mi = 1
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch) # Forward pass.
        loss = criterion(y_pred.squeeze(), y_batch)
        # For validation.
        y_pred_val = model(X_valid)
        loss_val = criterion(y_pred_val.squeeze(), Y_valid)
        print('Epoch {}, Batch: {}, Train loss: {}, Validation loss: {}'.format(epoch+1, mi, loss.item(), loss_val.item()))
        loss.backward() # Backward pass.
        optimizer.step()
        train_loss.append(loss.cpu().data.numpy())
        val_loss.append(loss_val.cpu().data.numpy())
        train_acc.append(compute_accuracy(y_pred, y_batch))
        val_acc.append(compute_accuracy(y_pred_val, Y_valid))
        mi += 1
# We plot the loss curves.
fig, ax = plt.subplots()
ax.plot(train_loss, label='Train loss')
ax.plot(val_loss, 'r', label='Validation loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
# We plot the accuracy curves.
fig2, ax2 = plt.subplots()
ax2.plot(train_acc, label='Train accuracy')
ax2.plot(val_acc, 'r', label='Validation accuracy')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
plt.show()


# ---------------- #
# MODEL EVALUATION #
# ---------------- #
model.eval()
y_pred = model(X_test)
acc = compute_accuracy(y_pred, Y_test)
print('Test accuracy: ' + str(acc) + '%')