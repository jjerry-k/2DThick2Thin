import tensorflow as tf

def Custom_MSE(y_true, y_pred):
    Err = y_true - y_pred
    Square = tf.square(Err)
    Mean = tf.reduce_mean(Square, axis = [1, 2, 3])
    return Mean

def Custom_RMSE(y_true, y_pred):
    Err = y_true - y_pred
    Square = tf.square(Err)
    Mean = tf.sqrt(tf.reduce_mean(Square, axis = [1, 2, 3]))
    return Mean

def Custom_SSIM(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    b, h, w, c = y_true.get_shape()
    tmp_true = tf.reshape(tf.transpose(y_true, [0, 3, 1, 2]), [b*c, h, w, 1])
    tmp_pred = tf.reshape(tf.transpose(y_pred, [0, 3, 1, 2]), [b*c, h, w, 1])
    true_max = tf.reduce_max(y_true)
    
    
    f1 = lambda: tf.constant(1)
    f2 = lambda: tf.constant(255)
    f3 = lambda: tf.constant(65535)
    max_val = tf.case({tf.greater(true_max, 255) : f3, tf.greater(true_max, 1) : f2},
                      default=f1, exclusive=False)
    
    output = tf.image.ssim(tmp_true, tmp_pred, max_val)
    output = tf.reshape(output, [b, c])
    return 1-tf.reduce_mean(output, axis=1)

def Custom_MS_SSIM(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    b, h, w, c = y_true.get_shape()
    tmp_true = tf.reshape(tf.transpose(y_true, [0, 3, 1, 2]), [b*c, h, w, 1])
    tmp_pred = tf.reshape(tf.transpose(y_pred, [0, 3, 1, 2]), [b*c, h, w, 1])
    true_max = tf.reduce_max(y_true)
    
    f1 = lambda: tf.constant(1)
    f2 = lambda: tf.constant(255)
    f3 = lambda: tf.constant(65535)
    max_val = tf.case([(tf.greater(true_max, 255), f3), (tf.greater(true_max, 1), f2)], default=f1)
    
    output = tf.image.ssim_multiscale(tmp_true, tmp_pred, max_val, [0.208, 0.589, 0.203])
    output = tf.reshape(output, [b, c])
    return 1-tf.reduce_mean(output, axis=1)
    
def mutual_information_single(hist2d):
    tmp = tf.cast(hist2d, dtype='float64')
    pxy = tmp / tf.reduce_sum(tmp)
    px = tf.reduce_sum(pxy, axis=1)
    py = tf.reduce_sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = tf.greater(pxy, 0)
    return tf.reduce_sum(tf.boolean_mask(pxy, nzs) * tf.log(tf.boolean_mask(pxy, nzs) / tf.boolean_mask(px_py, nzs)))

def tf_joint_histogram(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    
    vmax = 255
    b, h, w, c = y_true.get_shape()
    
    
    # Intensity Scaling
    max_int = tf.reduce_max(y_true, axis = [1,2], keepdims=True)
    tmp_true = tf.round(y_true / max_int * vmax)
    tmp_pred = tf.round(y_pred / max_int * vmax)
    
    
    # [batch, height, width, channel]
    # -> [batch, height * width, channel]
    # -> [batch, channel, height * width]
    flat_true = tf.transpose(tf.reshape(tmp_true, [b, h*w, c]), [0, 2, 1])
    flat_true = tf.reshape(flat_true, [b*c, h*w])
    flat_pred = tf.transpose(tf.reshape(tmp_pred, [b, h*w, c]), [0, 2, 1])
    flat_pred = tf.reshape(flat_pred, [b*c, h*w])
    
    output = (flat_pred * (vmax+1)) + (flat_true+1)
    # [b*c, 65536]
    output = tf.map_fn(lambda x : tf.histogram_fixed_width(x, value_range=[1, (vmax+1)**2], nbins=(vmax+1)**2), output)
    # [b, c, 256, 256] -> [b, 256, 256, c]
    output = tf.transpose(tf.reshape(output, [b, c, vmax+1, vmax+1]), [0, 2, 3, 1])
    return output, y_true, y_pred

def mutual_information(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    # [b, 256, 256, c]
    joint_histogram, _, _ = tf_joint_histogram(y_true, y_pred)
    b, h, w, c = joint_histogram.get_shape()
    
    # [b*c, 256, 256]
    joint_histogram = tf.reshape(tf.transpose(joint_histogram, [0, 3, 1, 2]), [b*c, h, w])
    
    output = tf.map_fn(lambda x : mutual_information_single(x), joint_histogram, dtype=tf.float64)
    output = tf.reshape(output, [b, c])
    return 1 - tf.reduce_mean(output, axis=1)