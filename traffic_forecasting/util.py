import numpy as np

class ZScoreNormalizer:
    
    def __init__(self):
        self.means = None
        self.std = None
        self.adapted = False
    
    def reset(self):
        self.means = None
        self.std = None
        self.adapted = False
    
    def adapt(self,X):
        self.means = np.mean(X,axis=0,keepdims=True)
        self.std = np.std(X,axis=0,keepdims=True)
        self.adapted = True
    
    def transform(self,X):
        assert(list(X.shape)[1:]==list(self.means.shape)[1:])
        assert(self.adapted)
        return (X-self.means)/self.std
    
    def inverse_transform(self,X):
        assert(list(X.shape)[1:]==list(self.means.shape)[1:])
        assert(self.adapted)
        return self.std*X + self.means
