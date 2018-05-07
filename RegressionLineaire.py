from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import minibatch_indexes,add_one_column

def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc

@decorator_vec
def mse(datax,datay,w,alpha):
    """ retourne la moyenne de l'erreur aux moindres carres """
    t = np.dot(w,datax.T)
    return ((datay - t)**2).mean()

@decorator_vec
def mse_g(datax,datay,w,alpha):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    p = datax.dot(w.T) - datay
    output = 2 * datax.T * p.T
    return output.mean(axis=1)


@decorator_vec
def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    return np.mean(np.maximum(0, - datay * datax.dot(w.T)))
    #return np.mean((- datay * datax.dot(w.T)).clip(min=0))
    
    
@decorator_vec
def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    t = datax.dot(w.T)
    positive = np.sign((- datay * t).clip(min=0))
    return positive.T.dot(datax) / len(datay)
    

class PlugInLineaire(object):
    def __init__(self,loss=mse,loss_g=mse_g,max_iter=1000,eps=0.01, alpha=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.alpha = alpha

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        datax = add_one_column(datax) # il s'agit d'ajouter le w0 dans le formule
        D = datax.shape[1]
        xinit = np.random.random((1,D))
        loss = lambda x: self.loss(datax, datay, x)
        loss_g = lambda x: self.loss_g(datax, datay, x)
        ### optimize
        x_histo = []
        f_histo = []
        
        batchsize = 32
        batches = minibatch_indexes(datax.shape[0],batchsize, shuffle=True)
        batch_count = batches.shape[0]
       
        grad_histo = []
        dernier = xinit
        for i in range(self.max_iter):
            batch = batches[i % batch_count]
            X = datax[batch]
            Y = datay[batch]
            fonc = lambda x: self.loss(X, Y, x, self.alpha)
            dfonc = lambda x: self.loss_g(X, Y, x, self.alpha)
            
            x_histo.append(dernier - (self.eps * dfonc(dernier)))
            f_histo.append(fonc(dernier))
            
            grad_histo.append(dfonc(dernier))
            dernier = x_histo[i]
        x_histo = np.array(x_histo)
        f_histo = np.array(f_histo)
        
        ###
        
        optimal_idx = f_histo.argmin()
        self.w = x_histo[optimal_idx].reshape(-1,1)
        
    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        datax = add_one_column(datax) # il s'agit d'ajouter le w0 dans le formule
        return np.array([np.sign(np.dot(self.w.T,x)) for x in datax]).flatten()

    def score(self,datax,datay):
        prediction = self.predict(datax)
        return np.count_nonzero(prediction == datay) / datax.shape[0]

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.show()


def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    #plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    perceptron = PlugInLineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plt.show(block=True)

 