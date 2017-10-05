class Solver:
    def __init__(self):
        pass
    
    def train(model,X,y,params):
        """
            model : All Weights in form of dictionary
                {
                    '1':{
                            'W':[]
                            'b':[]
                            ...
                        },
                    '2':{
                        'W':[],
                        'b':[],
                        ..
                    },...,
                    'n':{
                        'beta':[],
                        'gamma':[]
                    }
                }
            X: Input of dimension (N,C,H,W)
            y: Input of dimension (N,)
            params: parameters for gradient descent
                'alpha':learning_rate
                'method':gradient descent step
                'epoch' : Number of times to run gradient descent
                'batch_size': Number of training examples to sample
        """
        EPOCH = params.get('epoch',5000)
        BATCH_SIZE = params.get('batch_size',256)
        ALPHA = params.get('alpha',1e-5)
        METHOD = params.get('method','gd')
        J = []
        pass
            