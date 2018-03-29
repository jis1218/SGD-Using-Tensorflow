# coding: utf-8
'''
Created on 2018. 3. 28.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np
from SGD.UsingMNIST import UsingMNIST


if __name__ == '__main__':
    
    mnist = UsingMNIST()
 
    
    
#     x_data = np.float32(np.random.rand(2, 100)) # 2,100 행렬을 만든다. (0~1 값)
#     #print(x_data)
#     # 목표값을 만드는 식
#     y_data = np.dot([0.100, 0.200], x_data) + 0.300
#     
#     
#     # 선형 모델 정의
#     b = tf.Variable(tf.zeros([1])) # b변수 정의
#     W = tf.Variable(tf.random_uniform([1, 2], -1, 1)) # 파라미터에 -1이 들어가는 이유는 무엇인가?
#     y = tf.matmul(W, x_data) + b
#     
#     print(W)
#     print(b)
#     
#     # 손실함수와 학습함수 정의, 손실함수로는 평균 제곱 오차 사용
#     loss = tf.reduce_mean(tf.square(y-y_data))
#     optimizer = tf.train.GradientDescentOptimizer(0.5)
#     
#     # 손실함수의 값이 최소가 되게 설정
#     train = optimizer.minimize(loss)
#     
#     init = tf.global_variables_initializer()
#     
#     
#     sess = tf.Session()
#     sess.run(init) #초기화 하지 않으면 Attempting to use uninitialized value Variable_1 이런 에러가 뜬다.
#     output = sess.run(W)
#     print(output)
#     
#     for step in range(0, 201):
#         sess.run(train)
#         if step % 20 ==0:
#             print (step, sess.run(W), sess.run(b))
    
    
    
    
    pass