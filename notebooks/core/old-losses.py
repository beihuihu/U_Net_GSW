# Author: Ankit Kariryaa, University of Bremen



import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def tversky(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits 
    :param ground_truth: the segmentation ground_truth  
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :return: the loss
    """  
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    
    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(p0 * g0)
    fp = alpha * tf.reduce_sum(p0 * g1)
    fn = beta * tf.reduce_sum(p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

def accuracy(y_true, y_pred):#Calculates how often predictions equal labels.
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))#Accuracy =  (TP + TN) / (TP + TN + FP+ FN) 

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)

def dice_loss(y_true, y_pred):
    """compute dice loss"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_t, y_pred)

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def mIoU(y_true, y_pred):#the mean Intersection-Over-Union metric.
    # IoU = TP / (TP + FP + FN)
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp)/(K.sum(tp)+K.sum(fp)+K.sum(fn))
    

def sensitivity(y_true, y_pred):#recall = TP / (TP + FN)
    """compute sensitivity (recall)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))