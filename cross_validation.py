
import numpy as np
from scipy.sparse import vstack

def divide_intervalles(X, N):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    return np.array_split(indexes, N)

def validation_croisee(X,Y,method, N, equal_sized=False):
    """
    
    """
    intervalles = divide_intervalles(X,N)
    indexes = []
    if equal_sized:
        permutations = np.array([(i,j)for i in range(N) for j in range(N) if i != j])
        np.random.shuffle(permutations)
        for j in range(N):
            train,test = permutations[j]
            indexes.append((intervalles[train], intervalles[test]))
    else:
        for i in range(N):
            train = np.array([], dtype=int)
            test = np.array([], dtype=int)
            for j in range(N):
                if i != j:
                    train = np.concatenate((train,intervalles[j]))
                else:
                    test = np.concatenate((test,intervalles[j]))
            indexes.append((train,test))
    
    results = []
    for train,test in indexes:
        method.fit(X[train,:], Y[train])
        score = method.score(X[test,:], Y[test])
        results.append(score)
    return np.array(results).mean()