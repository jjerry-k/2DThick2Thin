import numpy as np

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
    return 10 * np.log10((d_r**2)/mse) #20*np.log10(65535) - 10*np.log10(np.sqrt(MSE))

# def SSIM(x, y):
#     """
#     x, y : (batch, height, width, channel)
#     """
#     I_xy = 2*np.mean(x, axis=(1, 2))*np.mean(y, axis=(1, 2))