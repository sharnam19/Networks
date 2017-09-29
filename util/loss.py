import numpy as np

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
    if y is None:
        return scores
    
    loss = 0
    loss += np.sum(-np.log(scores[range(N),y]))/N
    
    offset = np.zeros_like(scores)
    offset[range(N),y]=1
    dx = (scores-offset)/N
    return loss,dx

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
    if y is None:
        return scores
    
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
    return loss,dx