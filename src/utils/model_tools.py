import torch

import utils.nets as nets
from utils.exceptions import ArchitectureError


def train(dataloader, model, loss_fn, optimizer, device, swap=False, swap_labels=[]) -> float:
    '''
        Model training loop. Performs a single epoch of model updates.
        
        * USAGE *
        Within a training loop of range(num_epochs).

        * PARAMETERS *
        dataloader: A torch.utils.data.DataLoader object
        model: A torch model which subclasses torch.nn.Module
        loss_fn: A torch loss function, such as torch.nn.CrossEntropyLoss
        optimizer: A torch.optim optimizer
        device: 'cuda' or 'cpu'

        * RETURNS *
        float: The model's average epoch loss 
    '''

    size = len(dataloader.dataset)
    train_loss = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        if swap:
            for i in range(len(y)):
                if y[i] == swap_labels[0]:
                    y[i] = swap_labels[1]
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)

        # Backpropagation
        
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test(dataloader, model, loss_fn, device, swap=False, swap_labels=[]) -> float:
    '''
        Model test loop. Performs a single epoch of model updates.

        * USAGE *
        Within a training loop of range(num_epochs) to perform epoch validation, or after training to perform testing.

        * PARAMETERS *
        dataloader: A torch.utils.data.DataLoader object
        model: A torch model which subclasses torch.nn.Module
        loss_fn: A torch loss function, such as torch.nn.CrossEntropyLoss
        optimizer: A torch.optim optimizer
        device: 'cuda' or 'cpu'

        * RETURNS *
        float: The average test loss
    '''

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if swap:
                for i in range(len(y)):
                    if y[i] == swap_labels[0]:
                        y[i] = swap_labels[1]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


def add_output_nodes(ckpt:str, num_new_outputs:int=1, arch:str='linear') -> torch.nn.Module:
    '''
        TODO: Add func and sig description.
    '''

    ckpt = torch.load(ckpt)

    if arch == 'linear':
        # new part of weight matrix
        new_weights = torch.randn(
            num_new_outputs, ckpt['output_layer.weight'].shape[1])
        # new part of bias vector
        new_biases = torch.randn(num_new_outputs)

        # updated output layer weights
        ckpt['output_layer.weight'] = torch.cat(
            [ckpt['output_layer.weight'], new_weights], dim=0)
        # updated output layer biases
        ckpt['output_layer.bias'] = torch.cat(
            [ckpt['output_layer.bias'], new_biases], dim=0)

        # updated class total
        num_outputs = ckpt['output_layer.weight'].shape[0]
        # flattened image size
        input_size = ckpt['input_layer.weight'].shape[1]
        
        print("input_size", input_size)
        print("num_outputs", num_outputs)

        new_model = nets.LinearFashionMNIST_alt(input_size, num_outputs)
        new_model.load_state_dict(ckpt)
    elif arch =='cnn':
        new_model = nets.CIFAR10Cnn(num_outputs)
    elif arch == 'vgg':
        pass
    else:
        raise ArchitectureError(arch)

    return new_model
