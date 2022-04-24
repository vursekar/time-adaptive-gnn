import numpy as np

def mae_losses(preds,labels,axes=(0,1)):
    return np.mean(np.abs(preds-labels),axis=axes)

def mape_losses(preds,labels,axes=(0,1)):
    return np.mean(np.abs(np.divide(preds-labels,labels,
                                    out=np.zeros_like(preds), 
                                    where=labels!=0)),axis=axes)

def mse_losses(preds,labels,axes=(0,1)):
    return np.mean(np.square(preds-labels),axis=axes)

def smape_losses(preds,labels,axes=(0,1)):
    return np.mean(np.abs(preds-labels)/(np.abs(preds) + np.abs(labels)),axis=axes)