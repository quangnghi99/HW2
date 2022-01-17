import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    degree = 6
    res = np.ones(x1.shape[0])
    for i in range(0,degree + 1):
        for j in range(0,degree + 1):
            res = np.column_stack((res, (x1 ** (i)) * (x2 ** j)))
    return res

