import numpy as np
from ImageProcessing import ImageProcessing
from sklearn.linear_model import Lasso

def filterExpressedPixels(vec):
    # take only first array becouse we have only one dimension
    return np.nonzero(vec)[0]

def estimate_patch(patch, dictionary, lamb):
    imp = ImageProcessing()
    # rend le vecteur de poids sur le dictionnaire qui approxime au mieux le patch 
    # (restreint aux pixels exprimés) en utilisant l’algorithme du LASSO.
    # lamb la pénalisation du LASSO (controls the tradeoff between the reconstruction error and the sparsity)

    patch_vector = imp.patch_to_vector(patch)  # vecteur restreint aux pixels exprimés
    normExpPixels = np.linalg.norm(patch_vector)
    
    Y = patch_vector.reshape(-1,1) # to make it column vector, where each line correspond one example
    indexes = filterExpressedPixels(Y)
    
    # for learning, use only nonzero examples
    Y_nonexp = Y[indexes].reshape(-1,1)
    X = dictionary[indexes, :]
    
    model = Lasso()
    model.fit(X, Y_nonexp) # coefficient sparse
    beta = model.predict(X)
    return beta
    # je pense LASSO il fait tous ça, c'est pas le peine de faire par le main
    #return np.argmin(np.square(np.linalg.norm(patch_vector - np.dot(dictionary, beta)), axis=(1,2)) + lamb * normExpPixels)