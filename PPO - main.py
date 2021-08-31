#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Proximal Policy Optimzization algorithm
#Split version
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import random
from collections import deque
from ou_noise import OUNoise
import actor_PPO_net
import critic_PPO_net
import Environment as En
import os
from timeit import default_timer as timer   


reward_path = 'Monitoring/reward/'
if not os.path.exists(reward_path): os.makedirs(reward_path)

Noise_path = 'Monitoring/Noise/'
if not os.path.exists(Noise_path): os.makedirs(Noise_path)
    
Loss_path = 'Monitoring/Loss/'
if not os.path.exists(Loss_path): os.makedirs(Loss_path)


environment = En.env() 
_ = environment.reset()

alpha_critic = 0.9 

#Input & Output
state_nums = environment.state_num() 
action_nums = environment.action_setting() 

input_size_state= state_nums
input_size_action = state_nums
output_size_critic = 1 
output_size_actor =  action_nums


#Reinforcement learning parmeter
dis = 0.99 
buffer_memory = 50000 #Replay memory에 몇개를 넣을 것인가? (Buffer)
alpha_critic = 1.0
exploration_noise = OUNoise(input_size_action)
epsilon = 0.2 # for PPO CLIPPING


def train(main_actor, main_critic, train_batch, batch_size):

    #make empty stack
    state_stack = np.empty(0);  action_stack = np.empty(0)
    Q_target_stack = np.empty(0); Q_old_stack = np.empty(0)
    state_stack = np.reshape(state_stack, (0, input_size_state))
    action_stack = np.reshape(action_stack, (0, output_size_actor))
    Q_target_stack = np.reshape(Q_target_stack, (0, output_size_critic))
    Q_old_stack = np.reshape(Q_old_stack, (0, output_size_critic))    
    
    
    
    #----------------------------- Actor train -----------------------------#
    # Actor train part
    
    Loss_actor = 0.0
    
    for state, action, reward, next_state, done, old_log_prob in train_batch:
        
        #print(np.mean(action))
      
        #Old Q
        Q_old = main_critic.predict(state, action)         
        
        #make V_old
        V_old = Q_old - reward
      
        actor_loss, _  = main_actor.update(main_critic, state, next_state, action, old_log_prob, V_old, reward) #, main_actor   
        
        Loss_actor += actor_loss/len(train_batch)
    

    #----------------------------- Critic train -----------------------------#
    #Apprioxiomate TD error (Advantage)

    
    # Critic train part
    for state, action, reward, next_state, done, old_log_prob in train_batch:

        #Old Q
        Q_old = main_critic.predict(state, action)           
        
        #PPO - on-policy (main_actor)
        next_action, _ = main_actor.predict(next_state) 
        #next_action하고 log_prob반환하니까
        
        #target_Q
        Q_target = Q_old + alpha_critic*(reward + dis*(main_critic.predict(next_state, next_action)) - Q_old)     
 
        
        #Stacking process
        state_stack = np.vstack([state_stack, state])
        Q_target_stack = np.vstack([Q_target_stack, Q_target])
        Q_old_stack = np.vstack([Q_old_stack, Q_old])
        action_stack = np.vstack([action_stack, action])


    #Update critic model 
    Critic_loss, _  = main_critic.update(state_stack, action_stack, Q_target_stack)  

        
    return Critic_loss, Loss_actor




def main():

    st_step = 15 #action을 몇 time-step마다 취할 것인지에 대한 숫자
    state_step = 0
    record_frequency = 1
    step_deadline = 100
    update_freq = 1
    train_loop_epoch = 2 #PPO 
    max_episodes = 1000
    batch_size = 128
    buff_len = batch_size
    Loss_step = 0  

    
    # Replay buffer를 deque로 짠다. 
    buffer = deque() 

    reward_buffer = deque() #maxlen=100
    #reward_buffer또한 deque로 만들어서 마지막 100개까지 기억하도록 한다
    

    with tf.Session() as sess:
             
        #formation of network for actor net
        main_actor = actor_PPO_net.actor(sess, input_size_state, output_size_actor, output_size_critic, name="main_actor") 
       
        #formation of network for critic net (first error NameError - input_size ciritic 등)
        main_critic = critic_PPO_net.critic(sess, input_size_state, input_size_action, output_size_critic, main_actor, name="main_critic") 
        #main_critic에 main_actor를 넣어줘서 연결시킴 Q(s,a) - Objective
        
        _ = main_critic.initialization_c(name ="main_critic")
        _ = main_actor.initialization_a(main_critic.Objective, name ="main_actor")
        
        sess.run(tf.global_variables_initializer()) 
        print("initialization complete")    


        for episode in range(0, max_episodes+1):
            
            print("Episode : {} start ".format(episode))
        
            done = False
            
            ##################### environment로부터 state를 받아온다 (observation) ###############
            state = environment.reset() #envrionment로부터 state를 가져온다. (초기 state)
            exploration_noise.reset()       
            
            reward_graph = 0

            #Noise 그래프 그리기        
            
            reward_record = open(reward_path + "reward.plt" , 'a', encoding='utf-8', newline='') 
            noise_record = open(Noise_path + "noise, episode{}.plt" .format(episode), 'a', encoding='utf-8', newline='')
            state_reward_record = open(reward_path + "state_reward, episode{}.plt" .format(episode), 'a', encoding='utf-8', newline='')
            Loss_record = open(Loss_path + "Loss.plt".format(episode), 'a', encoding='utf-8', newline='')
            
            if episode ==0: 
                noise_record.write('VARIABLES = "state_step", "noise" \n') 
                state_reward_record.write('VARIABLES = "state_step", "avg_reward" \n')  
                reward_record.write('VARIABLES = "Episode", "Reward" \n') 
                Loss_record.write('VARIABLES = "state_step", "Loss" \n')
            
            
            while not done == True:

                Noise = 0.5*exploration_noise.noise() # 매 step마다 Normal distribution에서 임의로 추출한다.
                      
                #------------------------Stochastic policy part ------------------------#

                action, log_prob = main_actor.predict(state)
    
                #action_noise = action + (Noise) #with N(Noise) #이
                
                #noise를 입력한다
                noise_record.write("%d %f \n" %(state_step ,np.mean(Noise)))
                
                #Noise 조절이 좀 필요하다 step loop 안으로 들어와야 계속 변할수 있다. 
                #action_noise = np.reshape(action_noise, (input_size_critic_a))
                                
                # Get new state and reward from environment  
                next_state, reward, done, record = environment.simulation(state, action, st_step, step_deadline, episode)
                
                #한 Episode에서 순간 reward를 기록하기 위함
                state_reward_record.write("%d %f \n" %(state_step ,reward))
                

                #한 step의 reward씩 계속 reward_graph에 쌓는다. summation of reward
                reward_graph = reward + reward_graph

                #----------------------------- Replay buffer -------------------------#
                buffer.append((state, action, reward, next_state, done, log_prob))
                
                if len(buffer) > buffer_memory:
                    buffer.popleft()         
                    
                #--------------------------- Learning part ----------------------------#
                if len(buffer) > buff_len and state_step % update_freq == 0:

                    loss_avg = 0
                    
                    for _ in range(train_loop_epoch): #PPO에서는 train_loop_epoch을 활성화 시킨다.
                        
                        minibatch = random.sample(buffer, batch_size) 
                        minibatch = list(minibatch)
                        
                        #critic update start
                        loss_critic, loss_actor = train(main_actor, main_critic, minibatch, batch_size)
                        
                        loss_avg = loss_critic/train_loop_epoch +loss_avg

                    print("Loss for critic is : {}".format(loss_avg)) 
                    Loss_record.write("%d %f \n" %(Loss_step ,loss_avg))
                               
                #---------------------------------------------------------------------#       
               
                state = next_state
                
                state_step += 1
                Loss_step += 1
                #print("step num : {}".format(step))
                
                # break part
                if done==True or state_step == step_deadline:
                    break
               
            
            reward_graph = reward_graph/state_step
            
            state_step = 0 #state_step zero            
            
            #plt file로 reward graph 저장
            reward_record.write("%d %f \n" %(episode , reward_graph))
            
            noise_record.close()
            state_reward_record.close()
            Loss_record.close()
            reward_record.close()


if __name__ == "__main__":
    
    main()
    
    print("All process is finished!")


