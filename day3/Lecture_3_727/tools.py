import pickle
import torch


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# return loss and accuracy
def calc_loss_accuracy(dataloader, model, criterion, device):
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            """
            The output.data is like
            tensor([[0.3184, 0.3639, 0.3177],
            [0.3125, 0.3600, 0.3275],
            [0.3180, 0.3460, 0.3361],
            [0.3230, 0.3510, 0.3260]], dtype=torch.float64),

            This is a 2-d variable (tensor).
            - 1st dimension: batch size: 1
            - 2nd dimension: number of classes: 3

            We should get the max one in the 2nd dimension, so dim=1
            """
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            # predicted_onehot = torch.eye(3)[predicted.squeeze()].to(device)  # This can change [0,1,2] to [[1,0,0], [0,1,0], [0,0,1]].

            total += labels.size(0)
            correct += (predicted == actual).sum().item()
    return loss / len(dataloader), correct / total


def get_no_params(model):
    nop = 0
    for param in list(model.parameters()):
        nn = 1
        for s in list(param.size()):
            nn = nn * s
        nop += nn
    return nop


def get_no_params2(model):
    nop = 0
    for name, param in model.named_parameters():
        if 'bias' in name:
            nop += param.numel()
        elif 'weight' in name:
            nop += param.numel()
    return nop


def count_parameters3(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)