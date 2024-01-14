# ###################################
# Group ID : 727
# Members : Weifan Zhang
# Date : April 29, 2023
# Lecture: 1 Feedforward neural networks
# Dependencies: torch, numpy, matplotlib, copy, pickle
# Python version: 3.11.2 64-bit
# Functionality: This script implements and trains a feedforward neural network that solves a speech classification task. It also tests the performance on a given data set
# ###################################

import numpy as np
import tools
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy

# fix the random seed to make the results reproducible
torch.manual_seed(0)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

# flatten the data from (n_samples, 101, 40) to (n_samples, 101 * 40)
# The shape of labes is (n_samples, 3),
# The input of the neural network should have 101 * 40 dimensions, and the output of the neural network should have 3 dimensions.
x_train = x_train.view(-1, 101 * 40)
x_valid = x_valid.view(-1, 101 * 40)
x_test = x_test.view(-1, 101 * 40)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

# use DataLoader to handle the time frames in the data
train_data = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_data, batch_size=1024, shuffle=True)

valid_data = TensorDataset(x_valid, y_valid)
validloader = DataLoader(valid_data, batch_size=1024, shuffle=True)

test_data = TensorDataset(x_test, y_test)
testloader = DataLoader(test_data, batch_size=1024, shuffle=False)


# define our neural network
class MyNet(torch.nn.Module):

    def __init__(self, n_hidden_layers, hidden_layer_units, input_dim,
                 output_dim, activation):
        super(MyNet, self).__init__()

        self.input = torch.nn.Linear(
            input_dim, hidden_layer_units, dtype=torch.float64
        )  # The data type in the weight matrix of the function should be the same with our data type. The default is torch.float32, so we should change it to our torch.float64

        self.hidden_layers = torch.nn.Sequential()
        for i in range(n_hidden_layers):
            self.hidden_layers.add_module(
                f"hidden{i}",
                torch.nn.Linear(hidden_layer_units,
                                hidden_layer_units,
                                dtype=torch.float64))
            self.hidden_layers.add_module(f"activation{i}", activation)

        self.output = torch.nn.Linear(hidden_layer_units,
                                      output_dim,
                                      dtype=torch.float64)

    def forward(self, x):
        # print(self.input.weight.shape)
        # print(x.shape)

        x = self.input(x)
        x = self.hidden_layers(x)
        x = torch.softmax(
            self.output(x), dim=1
        )  # dim=1 means the sum in each row is 1, dim=0 means the sum in each column is 1
        # According to ChatGPT, for multiclass classification problems, softmax is typically used as the output layer activation function.
        return x


# instantiate the neural network
model = MyNet(4, 128, 101 * 40, 3, torch.nn.ReLU())
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=0.00001)
criterion = torch.nn.CrossEntropyLoss(
)  # According to ChatGPT, for multiclass classification problems, cross-entropy is used as the training loss function.

# From ChatGPT:
"""
For binary classification problems, sigmoid is usually used as the output layer activation function, and binary cross-entropy is used as the training loss function.

For multiclass classification problems, softmax is typically used as the output layer activation function, and cross-entropy is used as the training loss function.

For regression problems, no output layer activation function is typically used, and an appropriate training loss function is chosen depending on the specific scenario, such as mean squared error (MSE) or mean absolute error (MAE).
"""

# train the model
num_epochs = 50

# for plot, 2 figures:
# 1. Training Loss and Validation Loss
# 2. Train Accuracy and Validation Accuracy
iter = 1
iterations = []
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # calculate the loss and accuracy for plotting
        train_loss, train_accuracy = tools.calc_loss_accuracy(
            trainloader, model, criterion)
        valid_loss, valid_accuracy = tools.calc_loss_accuracy(
            validloader, model, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        iterations.append(iter)

        iter += 1

    # valid the model
    valid_loss, valid_accuracy = tools.calc_loss_accuracy(
        validloader, model, criterion)

    # calculate the training accuracy
    _, train_accuracy = tools.calc_loss_accuracy(trainloader, model, criterion)

    print(
        'Epoch [%d], Training Loss: %.4f, Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy: %.4f'
        % (epoch + 1, running_loss / len(trainloader), valid_loss,
           train_accuracy, valid_accuracy))

# test the model
_, test_accuracy = tools.calc_loss_accuracy(testloader, model, criterion)
print('Test accuracy: %.4f' % (test_accuracy))

# Plot the training and validation losses, as well as the training and validation accuracies, as a function of the training iteration.

plt.rcParams.update({'font.size': 16})

fig, ax1 = plt.subplots(figsize=(16, 8))

ax1.plot(iterations,
         train_losses,
         label='Training Loss',
         color='blue',
         linestyle='-',
         marker='o',
         markevery=10)

ax1.plot(iterations,
         valid_losses,
         label='Validation Loss',
         color='blue',
         linestyle='--',
         marker='^',
         markevery=10)
ax1.set_xlabel('Training Iteration')
ax1.set_ylabel('Loss Rate', color='blue')
ax1.tick_params(axis='y', colors='blue')

ax2 = ax1.twinx()
ax2.plot(iterations,
         train_accuracies,
         label='Train Accuracy',
         color='red',
         linestyle='-.',
         marker='s',
         markevery=10)
ax2.plot(iterations,
         valid_accuracies,
         label='Validation Accuracy',
         color='red',
         linestyle=':',
         marker='*',
         markevery=10)
ax2.set_ylabel('Accuracy Rate', color='red')

ax2.spines['right'].set_color('red')
ax2.spines['left'].set_color('blue')
ax2.tick_params(axis='y', colors='red')

x_ticks = copy.deepcopy(iterations)
for i in range(20):
    x_ticks.append(iterations[len(iterations) - 1] + i + 1)
ax2.set_xticks(x_ticks[::20])

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower center')

plt.title("Loss and Accuracy")
plt.show()
