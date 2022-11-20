from utils.feature_extractor import *
from torch import nn
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def get_sim_distr(model: nn.Module, dl: DataLoader, base_idx: [], new_idx: int):
    X, y = extract_features(model, dl, base_idx, new_idx)
    
    clf = LinearDiscriminantAnalysis().fit_transform(X,y)
    
    # TODO: fill in rest of cosine similarity stuff
    
    return clf
    

def extract_features(model: nn.Module, dl: DataLoader, base_idx: [], new_idx: int):
    X = []
    y = []
    
    class_subsets = generate_dls(dl, base_idx + [new_idx])

    # Yeah I know this is probably a dumb way to do this what can I say
    for class_idx in base_idx:
        for img, c in class_subsets[class_idx]:
            with torch.no_grad():
                feature = model(img)
            X.append(feature.numpy().reshape(-1))
            y.append(class_idx)
    
    new_class_ft = []

    for img, c in class_subsets[new_idx]:
        with torch.no_grad():
            feature = model(img)
        X.append(feature.numpy().reshape(-1))
        y.append(class_idx)
    
    return np.array(X), np.array(y)
    


def generate_dls(dl : DataLoader, classes: []) -> []:
    class_subsets = []

    for class_idx in classes:
        classes_idx = np.where((np.array(dl.targets) == class_idx))[0]
        class_subset = torch.utils.data.Subset(dl, classes_idx)
        class_subsets.append(class_subset)
    
    return class_subsets
