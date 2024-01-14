# ###################################
# Group ID : 727
# Members : Weifan Zhang
# Date : May 3, 2023
# Lecture: 3 recurrent neural networks
# Dependencies: torch, numpy, matplotlib, copy, pickle, threading, math, csv, datetime
# Python version: 3.11.2 64-bit
# Functionality: This script implements and trains a recurrent neural network that solves a speech classification task. It also tests the performance on a given data set.
# ###################################

import numpy as np
import tools
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
import threading
import math
import csv
import datetime

# fix the random seed to make the results reproducible
torch.manual_seed(0)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("device:", device)
if device.type == 'cuda':
    print('using GPU')
else:
    print('using CPU')

# read data from files
x_train = tools.load_data("data/X_train.p")
y_train = tools.load_data("data/Y_train.p")
x_valid = tools.load_data("data/X_valid.p")
y_valid = tools.load_data("data/Y_valid.p")
x_test = tools.load_data("data/X_test.p")
y_test = tools.load_data("data/Y_test.p")

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)
print(type(x_train[0][0][0]))
print(type(y_train[0][0]))
print(x_train[0][0][0])
print(y_train[0][0])

# normalize
train_mean = np.mean(x_train)
train_std = np.std(x_train)


def norm_func(data):
    return (data - train_mean) / train_std


x_train = norm_func(x_train)
x_valid = norm_func(x_valid)
x_test = norm_func(x_test)

# convert data to pytorch tensor
x_train = torch.tensor(x_train, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
x_valid = torch.tensor(x_valid, dtype=torch.float64)
y_valid = torch.tensor(y_valid, dtype=torch.float64)
x_test = torch.tensor(x_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)
print(type(x_train[0][0][0]))
print(x_train[0].shape)
print(type(y_train[0][0]))
print(x_train[0][0][0])
print(y_train[0][0])

# No need to reshape the data

# use DataLoader to handle the time frames in the data
loader_num_workers = 0
if device.type == 'cuda':
    loader_num_workers = 4
print("The number of data loader workers is {}.".format(loader_num_workers))
train_data = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_data,
                         batch_size=1024,
                         shuffle=True,
                         num_workers=loader_num_workers)

valid_data = TensorDataset(x_valid, y_valid)
validloader = DataLoader(valid_data,
                         batch_size=1024,
                         shuffle=True,
                         num_workers=loader_num_workers)

test_data = TensorDataset(x_test, y_test)
testloader = DataLoader(test_data,
                        batch_size=1024,
                        shuffle=False,
                        num_workers=loader_num_workers)

print(len(train_data))
print(len(valid_data))
print(len(test_data))

#######
"""
parameters of the model
"""
# Each sample of our data is a 101*40 matrix.
# In each sample, there are 101 time frames and the data in each time frame has 40 features.
input_size = 40

# parameters for GRU layers
gru1_hidden_size = 64
gru2_hidden_size = 64

# fully-connected layers
fc1_neurons = 128
fc2_neurons = 128

# other parameters
n_class = 3
train_epochs = 50


class MyRNN(torch.nn.Module):

    def __init__(self):
        super(MyRNN, self).__init__()
        # The shape of the input data is (n_sample, 101, 40). batch_first=True is to tell the GRU, the first dimension n_sample is the number of samples, the section dimension 101 is the number of time frames, and the third dimension 40 is the number of features of data at a time frame.
        self.gru1 = torch.nn.GRU(input_size=input_size,
                                 hidden_size=gru1_hidden_size,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False,
                                 dtype=torch.float64)
        self.gru2 = torch.nn.GRU(input_size=gru1_hidden_size,
                                 hidden_size=gru2_hidden_size,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False,
                                 dtype=torch.float64)
        self.fc1 = torch.nn.Linear(in_features=gru2_hidden_size,
                                   out_features=fc1_neurons,
                                   dtype=torch.float64)
        self.ac_fc1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=fc1_neurons,
                                   out_features=fc2_neurons,
                                   dtype=torch.float64)
        self.ac_fc2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(in_features=fc2_neurons,
                                      out_features=n_class,
                                      dtype=torch.float64)

    def forward(self, x):
        # x.shape torch.Size([n_sample, 101, 40])
        x, _ = self.gru1(x)
        # x.shape torch.Size([n_sample, 101, 64])
        x, _ = self.gru2(x)
        # x.shape torch.Size([n_sample, 101, 64])

        # # Eliminate the time dimension by average values. This is not what the assignment requires
        # x = torch.mean(x, dim=1)

        # Eliminate the time dimension by choosing the data at the last time frame. This is what the assignment requires
        x = x[:, -1, :]
        # print(x.shape)
        # x.shape torch.Size([n_sample, 64])
        x = self.fc1(x)
        x = self.ac_fc1(x)
        # x.shape torch.Size([n_sample, 128])
        x = self.fc2(x)
        x = self.ac_fc2(x)
        # x.shape torch.Size([n_sample, 128])
        x = self.output(x)
        # x.shape torch.Size([n_sample, 3])
        return torch.softmax(x, dim=1)


class model_runner():

    def __init__(self, optimizer_name, model, optimizer, criterion):
        self.optimizer_name = optimizer_name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.iterations = []
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracy = -1.0

    def run_model(self, num_epochs):
        train_start_time = datetime.datetime.now()

        # train the model
        self.model.train()

        # for plot, 2 figures:
        # 1. Training Loss and Validation Loss
        # 2. Train Accuracy and Validation Accuracy
        iter = 1

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # calculate the loss and accuracy for plotting
                train_loss, train_accuracy = tools.calc_loss_accuracy(
                    trainloader, self.model, self.criterion, device)
                valid_loss, valid_accuracy = tools.calc_loss_accuracy(
                    validloader, self.model, self.criterion, device)
                self.train_losses.append(train_loss)
                self.valid_losses.append(valid_loss)
                self.train_accuracies.append(train_accuracy)
                self.valid_accuracies.append(valid_accuracy)
                self.iterations.append(iter)

                iter += 1

            # valid the model
            valid_loss, valid_accuracy = tools.calc_loss_accuracy(
                validloader, self.model, self.criterion, device)

            # calculate the training accuracy
            _, train_accuracy = tools.calc_loss_accuracy(
                trainloader, self.model, self.criterion, device)

            print(
                'Optimizer [%s], Epoch [%d], Training Loss: %.4f, Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy: %.4f'
                %
                (self.optimizer_name, epoch + 1, running_loss /
                 len(trainloader), valid_loss, train_accuracy, valid_accuracy))

        train_end_time = datetime.datetime.now()
        train_duration = train_end_time - train_start_time

        # test the model
        test_start_time = datetime.datetime.now()
        self.model.eval()
        _, self.test_accuracy = tools.calc_loss_accuracy(
            testloader, self.model, self.criterion, device)
        test_end_time = datetime.datetime.now()
        test_duration = test_end_time - test_start_time
        print(
            'With Optimizer {}, Test accuracy: {:.4f}, Training time: {}, Test time: {}'
            .format(self.optimizer_name, self.test_accuracy, train_duration,
                    test_duration))

    def write_csv(self):
        with open(f'{self.optimizer_name}.csv',
                  'w',
                  newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "iterations", "train_losses", "valid_losses",
                "train_accuracies", "valid_accuracies"
            ])
            for i in range(len(self.iterations)):
                writer.writerow([
                    self.iterations[i], self.train_losses[i],
                    self.valid_losses[i], self.train_accuracies[i],
                    self.valid_accuracies[i]
                ])

    def read_csv(self):
        with open(f'{self.optimizer_name}.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                iteration, train_loss, valid_loss, train_accuracy, valid_accuracy = row
                self.iterations.append(int(iteration))
                self.train_losses.append(float(train_loss))
                self.valid_losses.append(float(valid_loss))
                self.train_accuracies.append(float(train_accuracy))
                self.valid_accuracies.append(float(valid_accuracy))


def plot_model(fig, fig_idx, total_fig, runner: model_runner):
    ax1 = fig.add_subplot(total_fig, 1, fig_idx)

    ax1.plot(runner.iterations,
             runner.train_losses,
             label='Training Loss',
             color='blue',
             linestyle='-',
             marker='o',
             markevery=10)

    ax1.plot(runner.iterations,
             runner.valid_losses,
             label='Validation Loss',
             color='blue',
             linestyle='--',
             marker='^',
             markevery=10)
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Loss Rate', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(runner.iterations,
             runner.train_accuracies,
             label='Train Accuracy',
             color='red',
             linestyle='-.',
             marker='s',
             markevery=10)
    ax2.plot(runner.iterations,
             runner.valid_accuracies,
             label='Validation Accuracy',
             color='red',
             linestyle=':',
             marker='*',
             markevery=10)
    ax2.set_ylabel('Accuracy Rate', color='red')

    ax2.spines['right'].set_color('red')
    ax2.spines['left'].set_color('blue')
    ax2.tick_params(axis='y', colors='red')

    x_ticks = copy.deepcopy(runner.iterations)
    for i in range(20):
        x_ticks.append(runner.iterations[len(runner.iterations) - 1] + i + 1)
    ax2.set_xticks(x_ticks[::20])
    ax2.tick_params(axis='x', rotation=45)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower center')
    ax1.set_title("Loss and Accuracy")


# do not execute the following code when this file is imported to another file
if __name__ == "__main__":
    # instantiate the Recurrent neural network with Adam
    model = MyRNN().to(device)
    # if we use GPU, calculate in parallel
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    runner = model_runner("Adam", model, optimizer, criterion)

    print("number of parameters in this RNN: {}".format(
        tools.get_no_params(model)))
    print("number of parameters2 in this RNN: {}".format(
        tools.get_no_params2(model)))
    print("number of parameters2 in this RNN: {}".format(
        tools.count_parameters3(model)))

    import time
    time.sleep(10000)

    # run the models
    print("start to run models")
    t_runner = threading.Thread(target=runner.run_model, args=(train_epochs, ))
    t_runner.start()
    t_runner.join()
    print("finish running models")

    # store the data in a csv file, because the training will cost too long time
    runner.write_csv()

    # Plot the training and validation losses, as well as the training and validation accuracies, as a function of the training iteration.
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(16, 8))

    plot_model(fig, 1, 1, runner)

    # fig.subplots_adjust(hspace=0.5)

    plt.show()
