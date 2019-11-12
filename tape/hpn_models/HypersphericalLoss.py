# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:56:01 2019

@author: Daniel
"""
import tensorflow as tf

def cosine_sim(a, b):
    """
        a: (batch1, batch2, ..., batchn, dim)
        b: (batch1, batch2, ..., batchn, dim)
        
        return: (batch1, batch2, ..., batchn)
            
    """
    
    dot_prod = tf.reduce_sum(tf.multiply(a, b) , axis=-1)
    norm_a = tf.norm(a, axis=-1)
    norm_b = tf.norm(b, axis=-1)
    return tf.math.divide(dot_prod, tf.multiply(norm_a,norm_b))

def pairwise_cosine_distance(a,b):
    """
        a: (m, dim)
        b: (n, dim)
        
        return: (m, n)
    """
    normalize_a = tf.nn.l2_normalize(a, axis=-1)
    normalize_b = tf.nn.l2_normalize(b, axis=-1)
    return 1 - tf.matmul(normalize_a, normalize_b, transpose_b = True)

def get_fake_prototypes(preds_l, prototypes_l, preds_u):
    """
    Returns fake prototypes as a weighted average of labels in the labelled subset. 
    Weights are a softmax of cosine distance. 
    
    Args:
        preds_l: (L, embed_dim)
        prototypes: (L, embed_dim)
        preds_u: (U, embed_dim)
    """
    raise Exception("Not implemented yet")
    
    distances = pairwise_cosine_distance(preds_u, preds_l) # (U, L)
    fake_prototypes = tf.matmul(tf.nn.softmax(distances, axis=-1), prototypes_l)
    return fake_prototypes

def make_hyperspherical_classification_loss(prototypes):
    """
    Args:
        prototypes: (num_classes, embed_dim)
    """
    
    def _get_prototypes(labels):
        return tf.tensordot(labels, prototypes, [[1], [0]])
    
    def loss(preds, labels):
        """
        Args:
            preds: (Batch, embed_dim)
            labels: (Batch, num_classes)
        """
        targets = _get_prototypes(labels)
        return tf.reduce_mean(tf.square(1 - cosine_sim(preds, targets)))
    
    return loss    
    
def make_hyperspherical_regression_loss(upper_prototypes, upper_bound, lower_bound):
    """
    Args:
        upper_prototypes: (label_dim, embed_dim)
            These should be close to orthogonal
        upper_bound: (label_dim). Upper bound for each label dimension.
        lower_bound: (label_dim). Lower bound for each label dimension.
    """    
    
    def _get_coefficients(labels):
        """
        Gets interpolation coefficients. 
        """
        return tf.math.divide(labels - lower_bound, upper_bound - lower_bound)
    
    def loss(preds, labels):
        """
        Args:
            preds: (Batch, embed_dim)
            labels: (Batch, label_dim)
        """
        coefficients = _get_coefficients(labels)
        return tf.reduce_mean(tf.square(coefficients - cosine_sim(preds, upper_prototypes)))
    
    return loss

        
        
            