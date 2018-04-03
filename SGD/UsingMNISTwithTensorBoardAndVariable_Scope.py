# coding: utf-8
'''
Created on 2018. 4. 3.

@author: Insup Jung
'''

import tensorflow as tf
from Dataset.mnist import * 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data

class UsingMNIST2(object):
    
    def __init__(self):
        learning_rate = 0.01
        training_epochs = 1000
        batch_size = 100
        display_step = 100
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [None, 784])
            
            t = tf.placeholder(tf.float32, [None, 10])
            
            output = self.inference(x)
            cost = self.loss(output, t)
            global_step = tf.Variable(0, name='global_step', trainable=False) #trainable이 False인 것은 이 변수가 학습될 수 없음을 나타낸다.
            train_op = self.training(cost, global_step, learning_rate)
            eval_op = self.evaluate(output, t)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter("logistic_logs/", graph_def=sess.graph_def)
            init_op = tf.global_variables_initializer() #tf.initialize_all_variables()는 deprecate 되었음
            sess.run(init_op)
            
            # 학습 주기
            for epoch in range(training_epochs):
                avg_cost = 0
                total_batch = int(mnist.train.num_examples/batch_size)
                for i in range(total_batch):                    
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x : batch_x, t : batch_y}
                    sess.run(train_op, feed_dict=feed_dict)
                    minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    avg_cost += minibatch_cost/total_batch
                
                if(epoch % display_step==0):
                    val_feed_dict = {x : mnist.validation.images, t : mnist.validation.labels}
                    accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                    print("Validation Error:", (1-accuracy))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
                   
        print("Optimization Finished!")
                   
        test_feed_dict = {x : mnist.test.images, t : mnist.test.labels}
        accuarcy = sess.run(eval_op, feed_dict = test_feed_dict )
        print("Test Accuracy:", accuracy)
            
    
    
    # inference는 추론이라는 뜻이다.
    def inference(self, x):
        init = tf.constant_initializer(value=0, dtype=tf.float32)
        W = tf.get_variable("W", [784, 10], initializer=init)
        b = tf.get_variable("b", [10], initializer=init)
        output = tf.nn.softmax(tf.matmul(x, W)+b)
        return output
    
    def loss(self, output, t):
        dot_product = t*tf.log(tf.clip_by_value(output, 1e-10, 1.0))
        cross_entropy = -tf.reduce_sum(dot_product, reduction_indices=1)
        loss = tf.reduce_mean(cross_entropy)
        return loss
    
    def evaluate(self, output, t):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    def training(self, cost, global_step, learning_rate):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step = global_step)
        return train_op
        