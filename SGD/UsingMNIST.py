# coding: utf-8
'''
Created on 2018. 3. 29.

@author: Insup Jung
'''

from Dataset.mnist import * 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data


import tensorflow as tf
import numpy as np

class UsingMNIST(object):
    '''
    classdocs
    '''
    def __init__(self):
        
        #(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        x = tf.placeholder("float", [None, 784]) #placeholder의 정확한 역할은 무엇인가?
        
        W1 = tf.Variable(tf.random_uniform([784,50], -0.4, 0.4)) #처음에 weight를 tf.zeros[784,50] 하는 것보다 값이 훨씬 잘 나옴, 초기값의 중요성
        b1 = tf.Variable(tf.zeros([50]))
        
        
        W2 = tf.Variable(tf.random_uniform([50,50], -0.4, 0.4))
        b2 = tf.Variable(tf.zeros([50]))
        
        W3 = tf.Variable(tf.random_uniform([50,10], -0.4, 0.4))
        b3 = tf.Variable(tf.zeros([10]))
        
        #y1 = tf.nn.softmax(tf.matmul(x, W1)+b1)
        y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
        
        y2 = tf.nn.relu(tf.matmul(y1,W2)+b2)
        
        y3 = tf.nn.softmax(tf.matmul(y2,W3)+b3)
        
        t = tf.placeholder(tf.float32, [None, 10])
         
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y3), reduction_indices=[1]))
         
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
             
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
         
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
             
        correct_prediction = tf.equal(tf.argmax(y3, 1), tf.argmax(t,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         
        print(sess.run(accuracy, feed_dict={x:mnist.test.images, t:mnist.test.labels})) #placeholder에 저장한 값을 run할 때 넣어준다. accuracy는 학습데이터가 아닌 테스트 데이터를 넣어준다.
#         
#         
#         
#         
#         print(np.shape(mnist.train.images)) 
#         print(mnist.train.labels)
        
        


        