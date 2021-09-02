#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############# DQN에 대한 공부 - Neural Network ############# 
import numpy as np
import tensorflow as tf


class actor:

    def __init__(self, session, input_size_actor_s, output_size_actor, output_size_critic, name="main"): 

        #객체 변수들에 대해서 다음과 같이 정의 한다. session, input_size, output_size, name 
        self.session = session
        self.input_size_actor_s = input_size_actor_s
        self.output_size_actor = output_size_actor
        #self.output_size_critic = output_size_critic
        self.net_name = name

        self.build_network() # class를 만들면 알아서 객체의 Neural network를 만들게 해 놓은거임.
        
     

        
    def build_network(self, h_size=128): 

        
        with tf.compat.v1.variable_scope(self.net_name, reuse=tf.compat.v1.AUTO_REUSE): 
            #변수의 범위는 변수가 효력을 미치는 영역을 의미한다 (변수의  scope) -namespace벗어나면 영향력이 없어진다.
            
            self.X_input = tf.compat.v1.placeholder(tf.float32, [None, self.input_size_actor_s], name="X_input")
            
            #for initialization (first layer)
            W1_S = [self.input_size_actor_s, h_size]
            W1_std = np.sqrt(1)/np.sqrt(np.prod(W1_S[:-1]))
            #Trainable parameter (first layer)
            self.W_a1 = tf.compat.v1.get_variable("W_a1", shape = W1_S, initializer = tf.random_normal_initializer(mean=0.0, stddev=W1_std, seed=None)) 
            self.B_a1 = tf.compat.v1.get_variable("B_a1", shape = [1,h_size], initializer = tf.initializers.zeros())
            
            #for initialization (second layer
            W2_S = [h_size, self.output_size_actor]
            W2_std = 10*np.sqrt(1)/np.sqrt(np.prod(W2_S[:-1]))           
            #Trainable parameter (second layer)
            self.W_a2 = tf.compat.v1.get_variable("W_a2", shape = W2_S, initializer = tf.random_normal_initializer(mean=0.0,stddev=W2_std, seed=None))   
            self.B_a2 = tf.compat.v1.get_variable("B_a2", shape = [1, self.output_size_actor], initializer = tf.initializers.zeros())  
            
            #연결층 형성 formation of network
            layer1 = tf.matmul(self.X_input*20, self.W_a1) + self.B_a1
            active1 = tf.nn.sigmoid(layer1)            
            
            layer2 = tf.matmul(active1, self.W_a2) + self.B_a2
            #if action is continuous --> likelihood, discrete --> probability        
            self.action_pred = layer2 #Action_nums만큼 개수가 추출된다 action list 수

            
            print("Actor_net connected")
            
            
    def initialization (self, Objective, l_rate=0.00001):
        
        self.Objective = Objective
            
        self.optimizer_a = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate, name = 'actor_adam')
        self.actor_vars = [self.W_a1, self.B_a1, self.W_a2, self.B_a2]
            
        self.train = self.optimizer_a.minimize(-self.Objective, var_list = self.actor_vars)
            

    def predict(self, state): 
        x = np.reshape(state, [1, self.input_size_actor_s])       
        return self.session.run(self.action_pred, feed_dict={self.X_input: x})
        #Tensor graph로 짜여있는 self.action_pred는 feed를 받고 (state) 그 값을 내보낸다.

    #Update (학습시키는거 데이터들을 받아서 self.session 실행시켜 돌려보낸다)        
    
    def update(self, main_critic, x_stack):

        #이상태로 그냥 node만 연결시키기는 힘든가보다 graph연결시킬때는 직접 variable을 연결시켜줘야 한다.
        feed = {self.X_input: x_stack, main_critic.input_critic_state: x_stack} 
           
        return self.session.run([self.train], feed_dict=feed)

    
    #state를 넣어줘야 operation들이 돌아가면서 Objective function들이 학습이 된다.