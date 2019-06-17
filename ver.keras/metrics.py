import numpy as np
from skimage.measure import compare_ssim as ssim

def MSE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.mean(np.square(y_true-y_pred), axis=(1, 2))

def RMSE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.sqrt(MSE(y_true, y_pred))

def RMSPE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.sqrt(np.mean(np.square((y_true - y_pred + 1)/(y_true+ 1)), axis=(1, 2)))

def PSNR(y_true, y_pred, d_r):
    '''
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    d_r : dynamic range
    '''
    mse = MSE(y_true, y_pred)
    return 10 * np.log10((d_r**2)/mse) #20*np.log10(65535) - 

def SSIM(y_true, y_pred, d_r):
    b, _, _, ch = y_true.shape
    output = np.zeros((b, ch))
    for b_idx in range(b):
        for ch_idx in range(ch):
            output[b_idx, ch_idx] = ssim(y_true[b_idx, ..., ch_idx], y_pred[b_idx, ..., ch_idx], data_range=d_r)
            
    return output