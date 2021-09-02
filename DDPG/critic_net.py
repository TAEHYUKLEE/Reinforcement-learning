#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############# DQN에 대한 공부 - Neural Network ############# 
import numpy as np
import tensorflow as tf


class critic:

    def __init__(self, session, input_size_critic_s, input_size_critic_a, output_size_critic, action_pred, name="main"): 
        #생선자를 만들어서 Class안의 Attribute (metod, members 등) 근데 이렇게 함수 받을때 :는 무슨 표시일까?

        #객체 변수들에 대해서 다음과 같이 정의 한다. session, input_size, output_size, name 
        self.session = session
        
        #이걸 action과 state로 다시 나눠서 해줘야한다
        self.input_size_critic_s = input_size_critic_s
        self.input_size_critic_a = input_size_critic_a
        self.output_size_critic = output_size_critic
        self.actor_action_pred = action_pred
        
        self.input_total_size_c = self.input_size_critic_s + self.input_size_critic_a
        
        self.net_name = name
        #self.action = tf.placeholder("float", [None, action_nums]) #원래는 grad Q X grad Policy 하려고 했던거
        #self.Q_grad = tf.gradients(self._Qpred, self.action)

        self.build_network()# class를 만들면 알아서 객체의 Neural network를 만들게 해 놓은거임.

        # 생성자는 return을 쓸수 없네.
        
    def build_network(self, h_size=128, l_rate=0.00001): #, h_size=10, l_rate=0.001

        
        with tf.compat.v1.variable_scope(self.net_name, reuse=tf.compat.v1.AUTO_REUSE): #정확이 이것은 무엇을 의미하는 것일까?
            #변수의 범위는 변수가 효력을 미치는 영역을 의미한다 (변수의  scope) -namespace벗어나면 영향력이 없어진다.
            
            self.input_critic_state = tf.compat.v1.placeholder(tf.float32, [None, self.input_size_critic_s], name="input_critic_state")
            self.input_critic_action = self.actor_action_pred #이렇게 이어주면 될듯하다. actor --> critic으로
            #나중에 self.input_critic_action으로 미분해줘야 함
            self.X_input = tf.concat([self.input_critic_state, self.input_critic_action], -1) #일렬로 만든다 reshape효과까지 합쳐서
            #나중에 그림으로 그려보자
            
            
            W1_S = [self.input_total_size_c, h_size]
            W1_std = np.sqrt(1)/np.sqrt(np.prod(W1_S[:-1]))
            #Trainable Parameter W_c1, B_c1
            self.W_c1 = tf.compat.v1.get_variable("W_c1", shape = W1_S, initializer = tf.random_normal_initializer(mean=0.0, stddev=W1_std, seed=None))
            self.B_c1 = tf.compat.v1.get_variable("B_c1", shape = [1,h_size], initializer = tf.initializers.zeros())
            
            W2_S = [h_size, self.output_size_critic]
            W2_std = 10*np.sqrt(1)/np.sqrt(np.prod(W2_S[:-1]))
            #Trainable Parameter W_c2, B_c2
            self.W_c2 = tf.compat.v1.get_variable("W_c2", shape = W2_S, initializer = tf.random_normal_initializer(mean=0.0,stddev=W2_std, seed=None))   
            self.B_c2 = tf.compat.v1.get_variable("B_c2", shape = [1, self.output_size_critic], initializer = tf.initializers.zeros())
            
            
            #연결층 만들기 
            layer1 = tf.matmul(self.X_input*20, self.W_c1) + self.B_c1
            active1 = tf.nn.relu(layer1)
            layer2 = tf.matmul(active1, self.W_c2) + self.B_c2
        
            self.Q_pred = layer2
              
            self.Y_target = tf.compat.v1.placeholder(shape=[None, self.output_size_critic], dtype = tf.float32)            
            
            #객체에 대한 Loss function을 만든다
            self.optimizer_c = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate, name = 'critic_adam')
            self.session.run(tf.compat.v1.variables_initializer(self.optimizer_c.variables()))
            
            self.loss = tf.reduce_mean(tf.square(self.Y_target - self.Q_pred))
            self.train = self.optimizer_c.minimize(self.loss)
            #아니 개 웃기네 ver1이라서 그냥 tf.get_variable등을 썼는데 여기서 왜 ver2처럼 ver1로 쓰고 있누...
            
            print("Critic_net connected")


    #Predict함수는 상태를 받아서 결과를 돌려달라가 된다.
    #신경회로망을 짜고 
    def predict(self, state, main_actor): # ,action, 이부분은 앞에서 action을 직접 이어보면서 더 이상 필요 없어졌다.
        input_state = np.reshape(state, [1, self.input_size_critic_s])
        print(input_state.shape)
        print(state.shape)
        print(state)
        print(self.input_critic_state.shape)
        #input_action = np.reshape(action, [1, self.input_size_critic_a]) #여기서 Policy로부터 받아온 action이고 critic으로 보내지면서 연결이 된다.
        return self.session.run(self.Q_pred, feed_dict={self.input_critic_state: input_state, main_actor.X_input: input_state })
        #self.input_critic_action:input_action
        
    #Update (학습시키는거 데이터들을 받아서 self.session 실행시켜 돌려보낸다)                   
    def update(self, x_state_stack, y_stack, main_actor):
        feed = {self.input_critic_state: x_state_stack, self.Y_target: y_stack, main_actor.X_input: x_state_stack}
        
        return self.session.run([self.loss, self.train], feed_dict=feed)
    
    '''
    Update부분 설명
    #self.loss에서는 loss만을 표기하고 self.train에서 학습이 일어난다.
    #feed로 self.X_input를 넣어주면 그 안의 graph에 있는 self.X_input에 그 값이 넣어진다.
    #X_input은 self.input_critic_state, self.input_critic_action로 이루어져 있고 이걸로 넣어줘야 한다 (안에서 합쳐져 있으므로)
    '''

