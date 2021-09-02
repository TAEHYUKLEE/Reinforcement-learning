#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############# DQN에 대한 공부 - Neural Network ############# 
import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, session, input_size, output_size, name="main"): #이 main부분은 어떻게 해야하는거지? str을 받는건데
        #생선자를 만들어서 Class안의 Attribute (metod, members 등) 근데 이렇게 함수 받을때 :는 무슨 표시일까?

        #객체 변수들에 대해서 다음과 같이 정의 한다. session, input_size, output_size, name 
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

        
    def _build_network(self, h_size=128, l_rate=0.0001): #, h_size=10, l_rate=0.001

        
        with tf.compat.v1.variable_scope(self.net_name, reuse=tf.compat.v1.AUTO_REUSE): #정확이 이것은 무엇을 의미하는 것일까?
            #변수의 범위는 변수가 효력을 미치는 영역을 의미한다 (변수의  scope) -namespace벗어나면 영향력이 없어진다.
            
            self._X = tf.compat.v1.placeholder(tf.float32, [None, self.input_size], name="input_x")
            W1_S = [self.input_size, h_size]
            W1_std = np.sqrt(1)/np.sqrt(np.prod(W1_S[:-1]))
            W1 = tf.compat.v1.get_variable("W1", shape = W1_S, initializer = tf.random_normal_initializer(mean=0.0, stddev=W1_std, seed=None))            
            B1 = tf.compat.v1.get_variable("B1", shape = [1,h_size], initializer = tf.initializers.zeros())
            
            layer1 = tf.nn.relu(tf.matmul(self._X*20, W1)) + B1
            W2_S = [h_size, self.output_size]
            W2_std = 10*np.sqrt(1)/np.sqrt(np.prod(W2_S[:-1]))
            W2 = tf.compat.v1.get_variable("W2", shape = W2_S, initializer = tf.random_normal_initializer(mean=0.0,stddev=W2_std, seed=None))   
            B2 = tf.compat.v1.get_variable("B2", shape = [1, self.output_size], initializer = tf.initializers.zeros())
            
            '''
            layer2 = tf.nn.relu(tf.matmul(self._X*20, W2)) + B2
            W3_S = [h_size, self.output_size]
            W3_std = 10*np.sqrt(1)/np.sqrt(np.prod(W3_S[:-1]))
            W3 = tf.compat.v1.get_variable("W3", shape = W2_S, initializer = tf.random_normal_initializer(mean=0.0,stddev=W3_std, seed=None))   
            B3 = tf.compat.v1.get_variable("B3", shape = [1, self.output_size], initializer = tf.initializers.zeros())
            '''
            #self.h_size 라고 되어 있어서 계속 에러 떴었던거야, 객체의 Attribute이 아니잖아
            #initializer = tf.contrib.layers.xavier_initializer()
        
            self._Qpred = tf.matmul(layer1, W2) + B2
              
            self._Y = tf.compat.v1.placeholder(shape=[None, self.output_size], dtype = tf.float32)            
            
            #객체에 대한 Loss function을 만든다
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
            
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)
            #아니 개 웃기네 ver1이라서 그냥 tf.get_variable등을 썼는데 여기서 왜 ver2처럼 ver1로 쓰고 있누...

    #Predict함수는 상태를 받아서 결과를 돌려달라가 된다.
    #신경회로망을 짜고 
    def predict(self, state): 

        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    #Update (학습시키는거 데이터들을 받아서 self.session 실행시켜 돌려보낸다)                   
    def update(self, x_stack, y_stack):
        feed = {self._X: x_stack, self._Y: y_stack}
        return self.session.run([self._loss, self._train], feed_dict=feed)

