import numpy as np

def update_weight(W,dW,params,regularization=False):
    """
        Input:
            W: Initial Weights
            dW: Gradient wrt Weights
            params: For Gradient Descent Method
           'alpha':learning_rate
           'reg' : regularization constant
           'reg_type':'None/L1'/'L2'
           'method': 'gd' | 'momentum' | 'adagrad'  |'rmsprop' | 'partial_adam' | 'adam'
           'epoch': Number of times to run gradient descent
           'momentum':  momentum (In momentum)
           'velocity' : gradient velocity for momentum method (In momentum)
           'grad_square' : sum of square of gradients (In rmsprop && adagrad)
           'offset'      : offset (about 1e-7 to avoid /0) (In rmsprop && adagrad && partial_adam && adam)
           'decay'       : value to decay grad_square by in rmsprop (In rmsprop)
           'beta1'       : decay value for firstmoment (In adam && partial_adam)
           'beta2'       : decay value for secondmoment (In adam && partial_adam)
           'firstmoment' : firstmoment value (In adam && partial_adam)
           'secondmoment': secondmoment value (In adam && partial_adam)
           't'           : update number (Only in Adam)
    """
    alpha = params.get('alpha',0)
    method = params.get('method',"gd")
    reg_type = params.get('reg_type','None')
    if reg_type is not 'None':
        reg = params.get('reg',0.0)
        if reg_type is 'L2':
            dW += 2*reg*W
        elif reg_type is 'L1':
            dW +=2*reg
    
    if method == "gd":
    
        W -= alpha*dW
    
    elif method == "momentum":
        
        momentum  = params.get("momentum",0.99)
        v = params.get("velocity",0.0)
        
        v = momentum*v + alpha*dW
        W -= v
        
        params["velocity"]=v
        
    elif method == "adagrad":
        
        gradSquare = params.get("grad_square",0.0)
        offset = params.get("offset",1e-7)
        
        gradSquare += np.square(dW)
        W -=alpha*dW/(np.sqrt(gradSquare)+offset)
        
        params["grad_square"]=gradSquare
    
    elif method == "rmsprop":
        
        decay = params.get("decay",0.9)
        gradSquare = params.get("grad_square",0.0)
        offset = params.get("offset",1e-7)
        
        gradSquare = decay*gradSquare + (1-decay)*np.square(dW)
        W -= alpha*dW/(np.sqrt(gradSquare)+offset)
        
        params["grad_square"]=gradSquare
    elif method == "partial_adam":
        
        beta1 = params.get("beta1",0.9)
        beta2 = params.get("beta2",0.999)
        mu1 = params.get("firstmoment",0.0)
        mu2 = params.get("secondmoment",0.0)
        offset = params.get("offset",1e-7)
        
        mu1 = beta1*mu1 + (1-beta1)*dW
        mu2 = beta2*mu2 + (1-beta2)*np.square(dW)
        W -= alpha*mu1/(np.sqrt(mu2)+offset)
        
        params["firstmoment"]=mu1
        params["secondmoment"]=mu2
        
    elif method == "adam":
        
        beta1 = params.get("beta1",0.9)
        beta2 = params.get("beta2",0.999)
        mu1 = params.get("firstmoment",0.0)
        mu2 = params.get("secondmoment",0.0)
        offset = params.get("offset",1e-7)
        t = params.get("t",0)
        t+=1
        mu1 = beta1*mu1 + (1-beta1)*dW
        mu2 = beta2*mu2 + (1-beta2)*np.square(dW)
        unbias1 = mu1/(1-beta1**t)
        unbias2 = mu2/(1-beta2**t)
        W -= alpha*unbias1/(np.sqrt(unbias2)+offset)
        
        
        params["firstmoment"]=mu1
        params["secondmoment"]=mu2
        params["t"]=t
    else:
        raise NotImplementedError