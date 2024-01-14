# ###################################
# Group ID : 727
# Members : Weifan Zhang
# Date : May 1, 2023
# Lecture: 2 Convolutional neural networks
# Dependencies: torch, numpy, matplotlib, copy, pickle, threading, math, csv, datetime
# Python version: 3.11.2 64-bit
# Functionality: This script implements and trains a convolutional neural network that solves a speech classification task. It also tests the performance on a given data set.
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

# If each data sample consists of X Y-dimensional metrices (X: number metrices, Y: number of dimensions in each matrix), we should use torch.nn.ConvYd(in_channels=X) to construct a convolutional neural network.
# In our data, each sample consists of 1 2-dimensional matrix, so we should use torch.nn.Conv2d(in_channels=1).
# To fit the torch.nn.Conv2d(in_channels=1), we need to reshape our data from (n_sample, 101, 40) to (n_sample, 1, 101, 40).
x_train = x_train.view(-1, 1, 101, 40)
x_valid = x_valid.view(-1, 1, 101, 40)
x_test = x_test.view(-1, 1, 101, 40)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

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
# parameters for convolutional layers
conv1_feather = 32  # feature maps of the first convolutional layer
conv2_feather = 16  # feature maps of the second convolutional layer
conv1_kernel_size = (5, 5)
conv1_stride = (1, 1)
conv2_kernel_size = (5, 5)
conv2_stride = (1, 1)

# parameters for max pooling
mp_conv1_kernel_size = (2, 2)
mp_conv1_stride = (2, 2)
mp_conv2_kernel_size = (2, 2)
mp_conv2_stride = (2, 2)

# parameters for fully-connected layers
fc1_neuron = 128  # neurons of the first fully-connected layer
fc2_neuron = 128  # neurons of the second fully-connected layer

# other parameters
n_matrix_per_sample = 1  # number of metrices in each data sample
n_class = 3  # number of classes (labels)
train_epochs = 50


def conv_mp_out_size(in_size, kernel, stride, padding):
    return math.floor((in_size - kernel + 2 * padding) / stride + 1)


def calculate_flattened_size():
    input_size = (x_train.shape[2], x_train.shape[3])
    print(f"input size is ({input_size[0]} x {input_size[1]})")

    conv1_out_size = (conv_mp_out_size(input_size[0], conv1_kernel_size[0],
                                       conv1_stride[0], 0),
                      conv_mp_out_size(input_size[1], conv1_kernel_size[1],
                                       conv1_stride[1], 0))
    print(f"conv1_out_size is ({conv1_out_size[0]} x {conv1_out_size[1]})")

    mp_conv1_out_size = (conv_mp_out_size(conv1_out_size[0],
                                          mp_conv1_kernel_size[0],
                                          mp_conv1_stride[0], 0),
                         conv_mp_out_size(conv1_out_size[1],
                                          mp_conv1_kernel_size[1],
                                          mp_conv1_stride[1], 0))
    print(
        f"mp_conv1_out_size is ({mp_conv1_out_size[0]} x {mp_conv1_out_size[1]})"
    )

    conv2_out_size = (conv_mp_out_size(mp_conv1_out_size[0],
                                       conv2_kernel_size[0], conv2_stride[0],
                                       0),
                      conv_mp_out_size(mp_conv1_out_size[1],
                                       conv2_kernel_size[1], conv2_stride[1],
                                       0))
    print(f"conv2_out_size is ({conv2_out_size[0]} x {conv2_out_size[1]})")

    mp_conv2_out_size = (conv_mp_out_size(conv2_out_size[0],
                                          mp_conv2_kernel_size[0],
                                          mp_conv2_stride[0], 0),
                         conv_mp_out_size(conv2_out_size[1],
                                          mp_conv2_kernel_size[1],
                                          mp_conv2_stride[1], 0))
    print(
        f"mp_conv2_out_size is ({mp_conv2_out_size[0]} x {mp_conv2_out_size[1]})"
    )
    print(f"conv2_feather is {conv2_feather}")

    return conv2_feather * mp_conv2_out_size[0] * mp_conv2_out_size[1]


flattened_size = calculate_flattened_size()
print(
    f"The number of the input features of the first fully-connected layer is {flattened_size}"
)


class MyCnn(torch.nn.Module):

    def __init__(self):
        super(MyCnn, self).__init__()
        """
        If each data sample consists of X Y-dimensional matrices (X: number of matrices, Y: number of dimensions in each matrix), we should use torch.nn.ConvYd(in_channels=X) to construct a convolutional neural network.
        In our data, each sample consists of 1 2-dimensional matrix, so we should use torch.nn.Conv2d(in_channels=1). 
        """

        # first convolution layer
        self.conv1 = torch.nn.Conv2d(in_channels=n_matrix_per_sample,
                                     out_channels=conv1_feather,
                                     kernel_size=conv1_kernel_size,
                                     stride=conv1_stride,
                                     dtype=torch.float64)
        self.mp_conv1 = torch.nn.MaxPool2d(kernel_size=mp_conv1_kernel_size,
                                           stride=mp_conv1_stride)
        self.relu_conv1 = torch.nn.ReLU()

        # second convolution layer
        self.conv2 = torch.nn.Conv2d(in_channels=conv1_feather,
                                     out_channels=conv2_feather,
                                     kernel_size=conv2_kernel_size,
                                     stride=conv2_stride,
                                     dtype=torch.float64)
        self.mp_conv2 = torch.nn.MaxPool2d(kernel_size=mp_conv2_kernel_size,
                                           stride=mp_conv2_stride)
        self.relu_conv2 = torch.nn.ReLU()

        # first fully-connected layer
        self.fc1 = torch.nn.Linear(in_features=flattened_size,
                                   out_features=fc1_neuron,
                                   dtype=torch.float64)
        self.relu_fc1 = torch.nn.ReLU()

        # second fully-connected layer
        self.fc2 = torch.nn.Linear(in_features=fc1_neuron,
                                   out_features=fc2_neuron,
                                   dtype=torch.float64)
        self.relu_fc2 = torch.nn.ReLU()

        # output layer
        self.output = torch.nn.Linear(in_features=fc2_neuron,
                                      out_features=n_class,
                                      dtype=torch.float64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.mp_conv1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.mp_conv2(x)

        # Flatten the data before fully-connected layers
        # Flatten the data into a vector with "conv2_feather * 22 * 7" elements.
        x = x.view(-1, flattened_size)

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = torch.softmax(self.output(x), dim=1)
        return x


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
    ax1.set_title("Loss and Accuracy with Optimizer {}".format(
        runner.optimizer_name))


# do not execute the following code when this file is imported to another file
if __name__ == "__main__":
    # instantiate the convolution neural network with SGD
    model_sgd = MyCnn().to(device)
    # if we use GPU, calculate in parallel
    if device.type == 'cuda':
        model_sgd = torch.nn.DataParallel(model_sgd)
    optimizer_sgd = torch.optim.SGD(params=model_sgd.parameters(),
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=0.00001)
    criterion_sgd = torch.nn.CrossEntropyLoss()
    runner_sgd = model_runner("SGD", model_sgd, optimizer_sgd, criterion_sgd)

    # instantiate the convolution neural network with Adam
    model_adam = MyCnn().to(device)
    # if we use GPU, calculate in parallel
    if device.type == 'cuda':
        model_adam = torch.nn.DataParallel(model_adam)
    optimizer_adam = torch.optim.Adam(params=model_adam.parameters())
    criterion_adam = torch.nn.CrossEntropyLoss()
    runner_adam = model_runner("Adam", model_adam, optimizer_adam,
                               criterion_adam)

    print("number of parameters in model_sgd: {}".format(
        tools.get_no_params(model_sgd)))
    print("number of parameters in model_adam: {}".format(
        tools.get_no_params(model_adam)))

    # import time;time.sleep(10000)

    # run the models
    print("start to run models")
    t_sgd = threading.Thread(target=runner_sgd.run_model,
                             args=(train_epochs, ))
    t_adam = threading.Thread(target=runner_adam.run_model,
                              args=(train_epochs, ))
    t_sgd.start()
    t_adam.start()
    t_sgd.join()
    t_adam.join()
    print("finish running models")

    # store the data in a csv file, because the training will cost too long time
    runner_sgd.write_csv()
    runner_adam.write_csv()

    # Plot the training and validation losses, as well as the training and validation accuracies, as a function of the training iteration.
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(16, 18))

    plot_model(fig, 1, 2, runner_sgd)
    plot_model(fig, 2, 2, runner_adam)

    fig.subplots_adjust(hspace=0.5)

    plt.show()
