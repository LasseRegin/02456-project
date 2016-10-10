# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 07:43:28 2016

@author: ec0di
"""
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
  