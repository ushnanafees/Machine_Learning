# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:59:52 2018

@author: Ushaan
"""
# Hello Tensorflow Program

import tensorflow as tf
hello = tf.constant("Hello Tensorflow!")
sess = tf.Session()                 # Run session method 1
print(sess.run(hello))


# Apply Multiplication on two Numbers (x,y)

x = tf.constant(4)
y = tf.constant(6)
result = tf.multiply(x,y)
with tf.Session() as sess:          # Run Session method 2
    print(sess.run(result))
