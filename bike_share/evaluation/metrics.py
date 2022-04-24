import numpy as np

def mae_losses(preds,labels):
    return np.mean(np.abs(preds-labels),axis=(0,2))[:,0]

def mape_losses(preds,labels):
    return np.mean(np.abs(np.divide(preds-labels,labels,
                                    out=np.zeros_like(preds), 
                                    where=labels!=0)),axis=(0,2))[:,0]

def rmse_losses(preds,labels):
    return np.mean(np.sqrt(np.mean(np.square(preds-labels),axis=2)),axis=0)[:,0]