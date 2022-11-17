import torch


def train(dataloader, model, loss_fn, optimizer, device) -> float:
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


def test(dataloader, model, loss_fn, device) -> float:
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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss
