import numpy as np

def window_average(x, fltr):
    window_shape = len(fltr)
    x = np.lib.stride_tricks.sliding_window_view(np.pad(x,(window_shape-1,0),constant_values=0),window_shape)
    return np.matmul(x,fltr)

def smooth_counts(X, filter_size=4, filter_type='even', var=1):
    if filter_type=='even':
        fltr = np.ones(filter_size)/filter_size
    elif filter_type=='gaussian':
        fltr = np.ones(filter_size)
        for i in range(filter_size):
            fltr[i] = np.exp(-i**2/var**2)
        fltr /= np.sum(fltr)
        fltr = fltr[::-1]

    print("Smoothing filter = ", fltr)
    X = np.apply_along_axis(lambda m: window_average(m, fltr), axis=1, arr=X)
    return X

def make_windows(X, window_length):
    num_samples = X.shape[1] - 2*window_length
    covariates = []
    labels = []

    for i in range(num_samples):
        covariates.append(X[:, i:i + window_length])
        labels.append(X[:, i + window_length:i + window_length + window_length])

    covariates = np.array(covariates, dtype=np.float32) # float64 takes too much space to store, hence using float32
    labels = np.array(labels, dtype=np.float32)
    return covariates, labels

def add_time_information(X, periods, keep_original_time=False):

    if len(periods)==0:
        return X

    X_time_feats = []
    for period in periods:
        X_time_feats.append(np.sin(2*np.pi*X[:,:,-1:]/period))
        X_time_feats.append(np.cos(2*np.pi*X[:,:,-1:]/period))

    if not keep_original_time:
        X = X[:,:,:2]

    X = np.dstack([X]+X_time_feats)
    return X

def seq_from_windows(X, horizon):
    return np.transpose(X[:,:,horizon],(1,0,-1))
