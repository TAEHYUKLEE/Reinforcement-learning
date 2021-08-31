#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow as tf2


def Smooth_l1_loss(labels, predictions, scope=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope):
        diff=tf.abs(labels-predictions)
        less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
        smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)#Same as above formula

        return tf.reduce_mean(smooth_l1_loss) #take the average



class critic:

    def __init__(self, session, input_size_critic_s, input_size_critic_a, output_size_critic, actor, name="main"): 

        self.session = session
        
        #이걸 action과 state로 다시 나눠서 해줘야한다
        self.input_size_critic_s = input_size_critic_s
        self.input_size_critic_a = input_size_critic_a
        self.output_size_critic = output_size_critic
        
        #-----------------state feed, action feed & actor connection ----------------#
        self.feed_state = tf.placeholder(tf.float32, [None, self.input_size_critic_s], name="input_critic_state")    
        self.feed_action = tf.placeholder(tf.float32, [None, self.input_size_critic_a], name="feed_action")
        self.action = actor.action  #직접 actor에서 가져온다.
        
        self.Input_size_t = self.input_size_critic_s + self.input_size_critic_a
        
        self.net_name = name

        
        #아래의 두개는 W1 B1 W2 B2 Trainable variable을 공유한다.
        print(name)
        self.Q_pred = self.build_network (self.feed_action, "Critic_net connected - for action_feed") #critic update할때 사용
        print(name)
        self.Objective = self.build_network (self.action, "Critic_net connected - for actor_feed") #actor update할때 사용.
        
    
    
    def build_network(self, action, sentence, h_1size = 128, h_2size = 128): 

        
        with tf.variable_scope(self.net_name, reuse=tf.compat.v1.AUTO_REUSE):

            self.input_critic_action = action

            self.X_input = tf.concat([self.feed_state, self.input_critic_action], -1) #일렬로 만든다 reshape효과까지 합쳐서

            
            W1_S = [self.Input_size_t , h_1size]
            #Trainable Parameter W_c1, B_c1
            self.W_c1 = tf.get_variable("W_c1", shape = W1_S, initializer = tf2.keras.initializers.GlorotNormal())
            self.B_c1 = tf.get_variable("B_c1", shape = [1,h_1size], initializer = tf.initializers.zeros())
                   
            
            W2_S = [h_1size, h_2size]
            #Trainable Parameter W_c2, B_c2
            self.W_c2 = tf.get_variable("W_c2", shape = W2_S, initializer = tf2.keras.initializers.GlorotNormal())   
            self.B_c2 = tf.get_variable("B_c2", shape = [1, h_2size], initializer = tf.initializers.zeros())
            
            
            W3_S = [h_2size, self.output_size_critic]
            #Trainable Parameter W_c1, B_c1 
            self.W_c3 = tf.get_variable("W_c3", shape = W3_S, initializer = tf2.keras.initializers.GlorotNormal())
            self.B_c3 = tf.get_variable("B_c3", shape = [1,self.output_size_critic], initializer = tf.initializers.zeros())            
            
            
            #Fully connected 
            layer1 = tf.matmul(self.X_input, self.W_c1) + self.B_c1
            #active1 = tf.nn.relu(layer1)
            active1 = tf.nn.relu(layer1, name=None)
            
            #tf.nn.leaky_relu(features, alpha=0.2, name=None)
            
            
            layer2 = tf.matmul(active1, self.W_c2) + self.B_c2
            #active2 = tf.nn.relu(layer2)
            active2 = tf.nn.relu(layer2, name=None)

            layer3 = tf.matmul(active2, self.W_c3) + self.B_c3
               
            Q_pred = 0.1*layer3
              
            self.Q_target = tf.compat.v1.placeholder(shape=[None, self.output_size_critic], dtype = tf.float32)          
             
            print(sentence)
        
        return Q_pred
        
        
        
    def initialization_c (self, name = "ops_name", l_rate=0.0002, B = 0.001):       
            #객체에 대한 Loss function을 만든다
            self.regular = tf.nn.l2_loss(self.W_c1)+tf.nn.l2_loss(self.W_c2)+tf.nn.l2_loss(self.W_c3)
            self.optimizer_c = tf.train.AdamOptimizer(learning_rate=l_rate, name = 'critic_adam')
            self.session.run(tf.variables_initializer(self.optimizer_c.variables()))
            
            #L2 Loss
            self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_pred)) + B*self.regular
            
            #L1 loss
            #self.loss = tf.reduce_mean(tf2.keras.losses.MAE(self.Q_target, self.Q_pred))+ B*self.regular
            
            #Smooth L1 loss
            #self.loss = Smooth_l1_loss(self.Q_target, self.Q_pred, scope=tf.GraphKeys.LOSSES)+ B*self.regular
            
            self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            
            self.train = self.optimizer_c.minimize(self.loss, var_list = self.critic_vars) 
            #[self.W_c1, self.B_c1, self.W_c2, self.B_c2, self.W_c3, self.B_c3]

            
            
    def predict(self, state, action): 
        input_state = np.reshape(state, [1, self.input_size_critic_s])
        input_action = np.reshape(action, [1, self.input_size_critic_a])

        return self.session.run(self.Q_pred, feed_dict={self.feed_state: input_state, self.feed_action: input_action})

    
                       
    def update(self, x_state_stack, x_action_stack, y_stack):
        feed = {self.feed_state: x_state_stack, self.feed_action: x_action_stack, self.Q_target: y_stack} 
        return self.session.run([self.loss, self.train], feed_dict=feed)
