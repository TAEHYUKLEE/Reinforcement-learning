#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow as tf2


class actor:

    def __init__(self, session, input_size_actor, output_size_actor, output_size_critic,  name="main"): 

        #객체 변수들에 대해서 다음과 같이 정의 한다. session, input_size, output_size, name 
        self.session = session
        self.input_size = input_size_actor
        self.output_size_actor = output_size_actor
        self.output_size_critic = output_size_critic
        self.net_name = name
        
        print(name)
        self.build_network() # class를 만들면 알아서 객체의 Neural network를 만들게 해 놓은거임.
        
        

        
    def build_network(self, h1_size=128): 

        
        with tf.variable_scope(self.net_name, reuse = tf.AUTO_REUSE): 
            #변수의 범위는 변수가 효력을 미치는 영역을 의미한다 (변수의  scope) -namespace벗어나면 영향력이 없어진다.
            
            self.X_input = tf.placeholder(tf.float32, [None, self.input_size], name="X_input")
            
            #for initialization (first layer)
            W1_S = [self.input_size, h1_size]
            #Trainable parameter (first layer)
            self.W_a1 = tf.get_variable("W_a1", shape = W1_S, initializer = tf2.keras.initializers.GlorotNormal()) 
            self.B_a1 = tf.get_variable("B_a1", shape = [1,h1_size], initializer = tf.initializers.zeros())
            
            
            #for initialization (mu layer)
            W_mu_S = [h1_size, self.output_size_actor]        
            #Trainable parameter (mu layer)
            self.W_mu = tf.get_variable("W_mu", shape = W_mu_S,initializer = tf2.keras.initializers.GlorotNormal())   
            self.B_mu = tf.get_variable("B_mu", shape = [1, self.output_size_actor], initializer = tf.initializers.zeros())  

            
            #for initialization (std layer)
            W_std_S = [h1_size, self.output_size_actor]      
            #Trainable parameter (std layer)
            self.W_std = tf.get_variable("W_std", shape = W_std_S, initializer = tf2.keras.initializers.GlorotNormal())   
            self.B_std = tf.get_variable("B_std", shape = [1, self.output_size_actor], initializer = tf.initializers.zeros())  
            
            
            #Action의 갯수만큼 mu, std가 나와야한다 --> Distribution
            
            #연결층 형성 formation of network
            layer1 = tf.matmul(self.X_input, self.W_a1) + self.B_a1
            active_mu = tf.nn.relu(layer1)   
            active_std = tf.nn.relu(layer1)

            #For actor
            mu = tf.matmul(active_mu, self.W_mu) + self.B_mu  
            #mu = 40.0*tf.nn.tanh(mu)
            
            #Std must be absolute value (if minus --> Error) - adding softplus active function last part, we avoid it
            std = tf.matmul(active_std, self.W_std) + self.B_std
            std = tf.math.abs(std)
            #std = tf.nn.softplus(std) #tf.nn.softplus

            self.action_mu = mu #mu를 예측한다
            self.action_std = std #std를 예측한다

            #mu, std를 반환해주는 것보다 actio, log_prob을 반환해주는게 낫다.
            dist = tfp.distributions.Normal(loc = self.action_mu, scale = self.action_std )
            self.action = tf.clip_by_value(dist.sample(), -1.0, 1.0)
            self.log_prob = dist.log_prob(self.action)


            print("Actor_net connected")
            
            
    def initialization_a (self, Q_value, name = "ops_name", l_rate=0.0001, B = 0.0001):
        
        #L2 Loss
        self.regular = tf.nn.l2_loss(self.W_a1)+tf.nn.l2_loss(self.W_mu)+tf.nn.l2_loss(self.W_std)
        
        self.Q_val = Q_value
        #Q_value가 Critic에다 직접 main_actor로 feed를 주어서 연결시켜 좋은 것.

        #---------------------------- State from Outside (main) -----------------------------#
        self.action_tensor = tf.ones( [1, self.output_size_actor],tf.float32, name="action")
        self.old_prob_tensor = tf.ones([1, self.output_size_actor],tf.float32, name="old_log_prob")
        self.V_old_tensor = tf.ones([1,1],tf.float32, name="V_old") 
        self.reward_tensor = tf.ones([1,1],tf.float32, name="reward")    

        #tensor to numpy 또 이렇게하면 이어지질 않는구나.
        action_np = self.action_tensor.eval(session=tf.Session())
        old_prob_np = self.old_prob_tensor.eval(session=tf.Session())
        V_old_np =  self.V_old_tensor.eval(session=tf.Session())
        reward =  self.reward_tensor.eval(session=tf.Session())
        
        #------------------------ get Liklihood ratio for Policy loss -----------------------#

        epsilon=0.2
        
        #mu, std = critic_model.predict(state_np) 
        dist_n = tfp.distributions.Normal(loc = self.action_mu, scale = self.action_std)
        new_log_prob = dist_n.log_prob(action_np)       
        ratio = (tf.math.exp(new_log_prob - old_prob_np))  # a/b == exp(log(a)-log(b))  #tensor라서 tf연산해줘야함.
        
        self.ratio = tf.reduce_mean(tf.math.exp(new_log_prob - old_prob_np))
              
        #get advantage function (Advantage Td(0)) for Critic loss
        self.advantage = tf.math.subtract(self.Q_val, V_old_np) #td_error   V_old_np
        
        #Surrogate policy update function (PPO)
        surrogate_1 = tf.math.multiply(ratio , self.advantage) #ratio는 Tensor로 들어가고 adavantage는 numpy로 들어가서 여기서 시간이 오래걸린거였음
        surrogate_2 = tf.math.multiply(tf.clip_by_value(ratio, 1-epsilon, 1+epsilon),  self.advantage) #tensor라서 tf연산해줘야 한다.     

        actor_loss =  tf.reduce_mean(tf.math.minimum(surrogate_1, surrogate_2)) 
        #actor_loss= actor_loss.eval(session=tf.Session()) #tensor to numpy (tensorflow version1)            
        #------------------------------------------------------------------------------------#        
        
        self.actor_loss = actor_loss        

           
        self.optimizer_a = tf.train.AdamOptimizer(learning_rate=l_rate, name = 'actor_adam')
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        #self.actor_vars = [self.W_a1, self.B_a1, self.W_mu, self.B_mu, self.W_std, self.B_std]
            
        self.train = self.optimizer_a.minimize(-actor_loss , var_list = self.actor_vars) #, var_list = self.actor_vars
        #2021-7-11 여기서는 함수를 minimize를 하는거지 함수값을 가지고 하는게 아니다보니 A(s,a)가 필요한거임 A(s=0.1, a=0.2)와 같은 value가 아닌.
        #Advantage function을 Objective function으로 하여 action parameter를 최대화 하는 방향으로 gradient ascent해야하는데 function이 아니니
        #Actor weight가 제자리 걸음하는거였음. Q로 짤수밖에 없다.
            

    def predict(self, state): 
        state = np.reshape(state, [1, self.input_size])       
        
        return self.session.run([self.action, self.log_prob], feed_dict={self.X_input: state})

 
    
    def update(self, critic_net, state, next_state, action, old_log_prob, V_old, reward):
        
        state = np.reshape(state, [1, self.input_size])
        V_old = np.reshape(V_old, [1,self.output_size_critic])
        reward = np.reshape(reward, [1,self.output_size_critic])
        old_log_prob = np.reshape(old_log_prob, [1,self.output_size_actor])
        action = np.reshape(action,[1,self.output_size_actor])
        
        
        feed = {self.old_prob_tensor: old_log_prob, self.action_tensor: action, self.reward_tensor:reward,\
                self.V_old_tensor: V_old, self.X_input:state, critic_net.feed_state:state} 

           
        return self.session.run([self.actor_loss, self.train], feed_dict=feed)
