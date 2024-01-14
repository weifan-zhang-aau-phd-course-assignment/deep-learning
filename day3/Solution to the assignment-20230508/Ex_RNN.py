#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:12:47 2020

@author: ivl@es.aau.dk
"""

# Assignment: Recurrent neural network (GRU).
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
class GRU(torch.nn.Module):
    
    def __init__(self, feature_dim, hidden_dim_gru, hidden_size, output_size):
        super(GRU, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim_gru = hidden_dim_gru
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = torch.nn.GRU(self.feature_dim, self.hidden_dim_gru, batch_first=True).cuda()
        self.gru2 = torch.nn.GRU(self.hidden_dim_gru, self.hidden_dim_gru, batch_first=True).cuda()
        self.fc3 = torch.nn.Linear(self.hidden_dim_gru, self.hidden_size).cuda()
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc5 = torch.nn.Linear(self.hidden_size, self.output_size).cuda()
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out, _ = self.gru(x)
        _, hc = self.gru2(out)
        out = self.fc3(hc[0,:,:])
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


# --------------------------- #
# ACCURACY COMPUTATION METHOD #
# --------------------------- #
def compute_accuracy(y_pred, y_labels):
    y_pred = y_pred.cpu().data.numpy()
    y_pred = y_pred.argmax(axis=1)
    acc = y_pred - y_labels.cpu().data.numpy()
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
model = GRU(no_qb,64,128,no_cl)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
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