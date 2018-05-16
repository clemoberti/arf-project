import numpy as np
from ImageProcessing import ImageProcessing
from sklearn.linear_model import Lasso

def filterExpressedPixels(vec):
    # take only first array becouse we have only one dimension
    return np.where(vec != -100)[0]

def estimate_patch(patch, dictionary, lamb):
    imp = ImageProcessing()
    # rend le vecteur de poids sur le dictionnaire qui approxime au mieux le patch 
    # (restreint aux pixels exprimés) en utilisant l’algorithme du LASSO.
    # lamb la pénalisation du LASSO (controls the tradeoff between the reconstruction error and the sparsity)
    
    Y = patch.reshape(-1,1) # to make it column vector, where each line correspond one example
    indexes = filterExpressedPixels(Y)
    
    # for learning, use only nonzero examples
    Y_nonexp = Y[indexes].reshape(-1,1)
    X = dictionary[indexes, :]
    
    model = Lasso(alpha=lamb, normalize=True)
    model = model.fit(X, Y_nonexp) # coefficient sparse
    #return model.predict(dictionary) # puis on predire les pixels du patch
    return np.array(model.coef_)
