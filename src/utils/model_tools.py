import torch

import utils.nets as nets
from utils.exceptions import ArchitectureError

import torchmetrics
from torchmetrics.classification import MulticlassRecall


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


def test(dataloader, model, loss_fn, device, swap=False, swap_labels=[], classes = 9) -> float:
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
    y_pred_list, targets = [], []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if swap:
                for i in range(len(y)):
                    if y[i] == swap_labels[0]:
                        y[i] = swap_labels[1]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #preds.append(pred)
            targets.append(y.numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            _, y_pred_tags = torch.max(pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    test_loss /= num_batches
    correct /= size
    
    recall = MulticlassRecall(classes)
    recall_val = recall(torch.FloatTensor(np.asarray(y_pred_list)), torch.IntTensor(np.asarray(targets)))

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Recall val: {recall_val:>8f} \n")

    return test_loss, np.asarray(y_pred_list), np.asarray(targets)


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
    elif arch == 'cnn-demo':
        # new part of weight matrix
        new_weights = torch.randn(
            num_new_outputs, ckpt['fc3.weight'].shape[1])
        # new part of bias vector
        new_biases = torch.randn(num_new_outputs)

        # updated output layer weights
        ckpt['fc3.weight'] = torch.cat(
            [ckpt['fc3.weight'], new_weights], dim=0)
        # updated output layer biases
        ckpt['fc3.bias'] = torch.cat(
            [ckpt['fc3.bias'], new_biases], dim=0)

        # updated class total
        num_outputs = ckpt['fc3.weight'].shape[0]

        new_model = nets.CNN_demo(num_outputs)
        new_model.load_state_dict(ckpt)
    elif arch == 'vgg':
        pass
    else:
        raise ArchitectureError(arch)

    return new_model

def get_recall_subsets(y_actual: [], y_preds: [], classes: [], total_classes_num: int) -> []:
    '''
    Generates recall values per epoch for a subset of classes.
    
    * USAGE *
    Use after training to get recall values (i.e. 'Old similar classes') for plotting improvements over time.

    * PARAMETERS *
    y_actual: Correct class labels, per batch, per epoch, collected from a test loop
    y_preds: Class label predictions, per batch, per epoch, collected from a test loop
    classes: The indices of classes which you would like to get recall values for
    total_classes_num: The total number of classes your model was trained on

    * RETURNS *
    list of float: List of recall values with length equal to number of epochs from the train loop
    '''

    recall_per_epoch = []
    recall = MulticlassRecall(total_classes_num)

    for e in range(len(y_actual)):
        y_per_epoch = np.asarray(y_actual[e]).flatten()
        preds_per_epoch = np.asarray(y_preds[e]).flatten()
    
        condition = y_per_epoch == classes[0]
        for i in range(1, len(classes)):
            condition |= y_per_epoch == classes[i]
    
        target_y = np.extract(condition, y_per_epoch)
        target_preds = np.extract(condition, preds_per_epoch)
    
        recall_val = recall(torch.IntTensor(target_preds), torch.IntTensor(target_y))
    
        recall_per_epoch.append(recall_val.item())
        
    return recall_per_epoch