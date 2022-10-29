import numpy as np


def create_index_arrays(dataset, indices=[]):
    '''
        Builds numpy arrays of dataset indices corresponding to
        dataset items which should be retained or excluded.

        * USAGE *
        To remove classes from a dataset.

        * PARAMETERS *
        dataset: A torch.utils.data.DataSet ot torchvision.datasets object
        indices: A list of integers representing class indices to remove

        * RETURNS *
        included dataset indices: np.array
        excluded dataset indices: np.array
    '''

    included = []
    excluded = []

    for idx, item in enumerate(dataset):
        if item[1] in indices:
            excluded.append(idx)
        else:
            included.append(idx)

    return np.array(included), np.array(excluded)
