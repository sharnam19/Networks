import numpy as np
import copy
def softmax_loss(x,y=None):
    """
        Input:
            x of shape (N,D)
            y of shape (N,). It is the class values
        Output:
            loss: Loss Value
            dx: Softmax Loss wrt the input. Same Shape as x
    """
    maximum = np.max(x,axis=1)
    shifted_x = x - maximum[:,np.newaxis]
    exp_x = np.exp(shifted_x)
    scores = exp_x/np.sum(exp_x,axis=1)[:,np.newaxis]
    N,D = x.shape
    predicted = np.argmax(scores,axis=1)
    if y is None:
        return predicted
    loss = 0
    loss += np.sum(-np.log(scores[range(N),y]))/N
    
    offset = np.zeros_like(scores)
    offset[range(N),y]=1
    dx = (scores-offset)/N
    return predicted,loss,dx

def svm_loss(x,y=None):
    """
        Input:
            x of shape (N,D)
            y of shape (N,). It is the class values
        Output:
            loss : loss value
            dx  : SVM Loss wrt input. Same shape as x
    """
    scores = x
    predicted = np.argmax(scores,axis=1)
    if y is None:
        return predicted
    
    N,D = x.shape
    correct_scores = scores[range(N),y][:,np.newaxis]
    margin = scores - correct_scores + 1
    scores[range(N),y]=0
    scores = np.maximum(0,scores)
    loss = np.sum(scores)/N
    
    ones = np.sign(scores)
    row_sum = np.sum(ones,axis=1)
    ones[range(N),y]=-row_sum
    dx = ones/N
    return predicted,loss,dx

def mse_loss(x,y=None):
    """
        Input:
            x of shape (N,) 
            y of shape (N,). It is continuous values
        Output:
            loss : loss value
            dx :gradient of RMSE Loss wrt input. Same shape as x
    """
    scores = x
    if y is None:
        return scores
    
    N = x.shape[0]
    diff = x-y
    loss = np.mean(np.square(diff))/2
    dx = diff/N
    return x,loss,dx
 
def cross_entropy_loss(x,y=None):
    """
        Input:
            x of shape (N,D)
            y of shape (N,D). It should be class values
        Output:
            loss : loss value
            dx : Gradient of Cross-Entropy Loss wrt input.Same shape as x
    """
    scores = x
    predicted = np.round(scores,0)
    if y is None:
        return predicted
    elif len(y.shape)==1:
        y = np.reshape(y,(-1,1))
        x = np.reshape(x,(-1,))
    N=x.shape[0]
    t= -y*np.log(scores)-(1-y)*np.log(1-scores)
    loss = np.sum(t)/(2*N)
    dx = -y/scores+(1-y)/(1-scores)
    dx /= 2*N
    return predicted,loss,dx
