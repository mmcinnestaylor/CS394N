from utils.feature_extractor import *
from torch import nn
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import math



def compute_similarity(a, b) -> float:
    d = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1/(1 + math.exp(d))


def get_similarity_mat(avg_actives):
    
    '''
    l = len(avg_actives)
    d = lambda vi, vj : 1 - (np.dot(vi, vj)/(np.linalg.norm(vi) * np.linalg.norm(vj)))
    s = lambda vi, vj : 1 / (1 + exp(d(vi,vj)))
    
    sim_mat = [[] * l] * l # how do I do this in numpy?
    
    for i in range(l): # + 1?
        for j in range(l):
            if i == j:
                sim_mat[i][j] = 0
            else:
                summ = 0
                summ = summ + s(avg_actives[i], avg_actives[k]) for k in range(j, len(avg_actives))
                sim_mat[i][j] = s(avg_actives[i], avg_actives[j]) / summ
    
    return sim_mat
    '''
    return

    

def get_lda_avgs(X, y, subset_size):
    trans_act = LinearDiscriminantAnalysis().fit_transform(X,y)
    
    # group the data by classes and get avg class activation
    class_splits_trans = []

    for i in range(1,int(len(trans_act)/subset_size) + 1):
        idx = subset_size * i
        class_splits_trans.append(trans_act[idx-subset_size:idx])
    
    avgs = np.mean(class_splits_trans, axis=1, dtype=np.float64)
    
    return avgs
    

def extract_features(model: nn.Module, dl: DataLoader, base_idx: [], new_idx: int):
    X = []
    y = []
    class_subsets, subset_size = generate_dls(dl, base_idx + [new_idx])

    # Yeah I know this is probably a dumb way to do this what can I say
    for class_idx in base_idx:
        for img, c in class_subsets[class_idx]:
            with torch.no_grad():
                feature = model(img)
            X.append(feature['input_layer'].numpy().flatten())
            y.append(class_idx)

    for img, c in class_subsets[new_idx]:
        with torch.no_grad():
            feature = model(img)
            
        X.append(feature['input_layer'].numpy().flatten())
        y.append(new_idx)
    
    return np.array(X), np.array(y), subset_size
    

def generate_dls(dl : DataLoader, classes: []): 
    class_subsets = []

    for class_idx in classes:
        # Struggling to work with our subsets, this works faster
        classes_idx = np.where((np.array(dl.targets) == class_idx))[0]
        class_subset = torch.utils.data.Subset(dl, classes_idx)
        class_subsets.append(class_subset)
        
    subset_size = len(classes_idx)
    
    return class_subsets, subset_size
