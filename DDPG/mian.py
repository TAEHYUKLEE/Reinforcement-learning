#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow.compat.v1 as tf
import random
from collections import deque
import critic_net
import actor_net
import Environment as En
#from typing import List

environment = En.env() #call environment
#환경을 부른다. (사실 Environment라는 객체를 만든다)

_ = environment.reset() #환경을 초기화한다.

alpha_critic = 0.9 #learning rate (based on Q)
#alpha_actor = 0.9

#Input & Output
state_nums = environment.state_num() #Q function은 (action, state) 이 두개에 의해 결정이 되므로, action까지 넣어줘야 한다.
action_nums = environment.action_setting() #하나의 action 구성 list 개수를 의미한다.

input_size_critic_s = state_nums
input_size_critic_a = action_nums #일단 network는 나눠서 줄거니까.
output_size_critic = 1 #DQN과 다르게 모든 action에 대한 Q를 계산하는 것이 아니라, Policy가 최적의 action을 선택해주기때문에 
                       #하나의 Q값이 생긴다 (여느 state에 대한 최적의 action에 대해서)
input_size_actor_s = state_nums
output_size_actor =  action_nums


#Reinforcement learning parmeter
dis = 0.99  
buffer_memory = 50000 #Replay memory에 몇개를 넣을 것인가? (Buffer)
batch_size = 100 #Mini batch size Buffer에서 몇개씩 batch로 만들어서 학습시킬 것인가?


def critic_train(main_critic, target_critic, main_actor, target_actor, train_batch):
   # 학습시킬 Network와 데이터 batch가 배달옴
    Q_old = np.zeros([1], dtype = np.float64) 
    Q_new = np.zeros([1], dtype = np.float64)
    
    x_action_stack = np.empty(0)
    x_state_stack = np.empty(0)
    y_stack = np.empty(0)
    
    x_state_stack = np.reshape(x_state_stack, (0, main_critic.input_size_critic_s))
    x_action_stack = np.reshape(x_action_stack, (0, main_critic.input_size_critic_a))
    y_stack = np.reshape(y_stack, (0, target_critic.output_size_critic)) #output_size_critic = 1로되어있다.

    for state, action_noise, reward, next_state, done in train_batch: #이 부분들 다시 한 번 보도록 하자 (3번째 Cell에 연습함)

        # Q = main_critic.predict(state, action) #이부분 state와 action을 같이 넣고 싶은데 문제가 생길듯.
        # state, action_noise같이 넣어도 됨 그렇게 critic을 바꾸어 놓았다. (원래는 np.reshape이용해서 한 번에 합쳐서 보냈었음)
        
        #Q_old (prediction)
        Q_old = main_critic.predict(state, main_actor) 
        
        #next_state_action또한 정해줘야 한다 - target policy를 이용하여 next_state에 대한걸 넣어 next_action을 유추.
        next_action = target_actor.predict(next_state) #target policy로 next_state에 대한 next_action을 받아온다.
        
        if done:
            Q_new = reward
        else:
            Q_new = Q_old + alpha_critic*(reward + dis*(target_critic.predict(next_state, target_actor)) - Q_old)
            
        y_stack = np.vstack([y_stack, Q_new])
        x_state_stack = np.vstack([x_state_stack, state]) #state를 학습시키는거지 Q를 학습시키는건 아니다.

        x_action_stack = np.vstack([x_action_stack, action_noise])
        
        #actor에 input을 같이 줘야한다.
        loss_critic, _ = main_critic.update(x_state_stack, y_stack, main_actor)
        
    return loss_critic, Q_old, Q_new




def actor_train(main_actor, target_actor, main_critic, train_batch):
    
    sess = tf.Session()
    x_stack_actor = np.zeros([input_size_actor_s], dtype = np.float64)
    x_stack_Q = np.zeros([input_size_critic_s], dtype = np.float64)
    
    for state, action, reward, next_state, done in train_batch: #이 부분들 다시 한 번 보도록 하자 (3번째 Cell에 연습함)

        #actor_var = main_actor.predict(state)
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_actor")

        x_stack = np.vstack([x_stack_actor, state]) 
        
        Objective_Q = main_critic.Q_pred
        
        _ = main_actor.update(main_critic, x_stack)
        
        #서로 연결되어 있어서 main_actor update하기 위해 main_critic에 X_input를 넣어줘야한다
        #InvalidArgumentError: You must feed a value for placeholder tensor 
        #'main_critic/input_critic_state' with dtype float and shape [?,180]
        
    return _   



def copy_var_ops(*, target_scope_name ="target", main_scope_name = "main"):

    op_holder = []

    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)
    
    
    for main_var, target_var in zip(main_vars, target_vars): 
        op_holder.append(target_var.assign(main_var.value()))
        #dest_var(tensor). assign
        
    return op_holder



def main():
    
    Q_old = np.empty(0)
    Q_new = np.empty(0)  
    st_step = 1 #action을 몇 time-step마다 취할 것인지에 대한 숫자
    state_step = 0
    record_frequency = 3
    step_deadline = 1000
    main_update_freq = 3
    target_update_frequency = 3
    #main이 target을 향해서 update되어가고 이후에 target_update가 이루어져야 하기때문에 main_freq < target_update가 되어야 한다.
    max_episodes = 100
    
    # Replay buffer를 deque로 짠다. 
    buffer = deque() 
    #Memory는 50000개까지 

    reward_buffer = deque() #maxlen=100
    #reward_buffer또한 deque로 만들어서 마지막 100개까지 기억하도록 한다
    
    reward_record = open("reward.plt" , 'w', encoding='utf-8', newline='') 
    reward_record.write('VARIABLES = "Episode", "Reward" \n') 
    #Reward를 기록하기 위함.
    
    with tf.Session() as sess:
        
        #formation of network for actor net
        main_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="main_actor") 
        #TypeError: __init__() missing 1 required positional argument: 'output_size_critic' (main_actor에서)
        target_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="target_actor")
        
        
        #formation of network for critic net (first error NameError - input_size ciritic 등)
        main_critic = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, main_actor.action_pred,  name="main_critic") 
        target_critic = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, target_actor.action_pred, name="target_critic")     
        #main_actor.action_pred를 줌으로써 이어줘 본다.
        
        _ = main_actor.initialization(main_critic.Q_pred)
        _ = target_actor.initialization(target_critic.Q_pred)
    
        sess.run(tf.global_variables_initializer()) #initializer <여기서 전체적으로 초기화해준다.>
        print("initialization complete")
        
        #Critic (copy)
        copy_ops_critic = copy_var_ops(target_scope_name="target_critic",main_scope_name="main_critic")
        sess.run(copy_ops_critic)
        
        #Policy (copy)
        copy_ops_actor = copy_var_ops(target_scope_name="target_actor", main_scope_name="main_actor")
        #sess.run(copy_ops_actor)
        


        for episode in range(0, max_episodes+1):
            
            print("Episode : {} start ".format(episode))
        
            done = False
            
            ##################### environment로부터 state를 받아온다 (observation) ###############
            state = environment.reset() #envrionment로부터 state를 가져온다. (초기 state)
            
            reward_graph = 0
            
            
            ############### 두개의 Neural network로 학습을 시키는 부분이다 ##########
               
            #정확히는 Episode 10이 끝난 시점에서 update하는 것이다.
            
            if episode > main_update_freq and episode % main_update_freq == 0: # train every 10 episodes
                print("update start")
                loss_avg = 0
                
                for _ in range(50):
                    #print("random_sample, step :{}" ,format(_)) #check complete
                    minibatch = random.sample(buffer, batch_size) 
                    minibatch = list(minibatch)
                    
                    #print("critic update start")
                    loss_critic, Q_old, Q_new= critic_train(main_critic, target_critic, main_actor, target_actor, minibatch)
                    
                    #print("actor update start")
                    _ = actor_train(main_actor, target_actor, main_critic, minibatch)
                    
                    loss_avg = loss_critic/50 +loss_avg
                    
                print("Loss for critic is : {}".format(loss_avg))   
                print("update end")
            ########################################################################
            
            if episode % record_frequency == 0:
                record = environment.record_start(episode)
                
                
                
            while not done: #이부분이 한 episode에서 state를 진행시키는 부분 actor critic을 이 부분에 넣어야함.
                
                #Noise 매 step마다 Noise의 정도는 작아지게 설정할 것이다.
                Noise = np.random.rand(1) # 매 step마다 Normal distribution에서 임의로 추출한다.
                N = Noise*0.0001 / ((state_step / 10) + 1) #갈수록 Noise는 작아지게
                
                ##################### deterministic policy에서 state를 주고 action을 뽑아온다 ###################
                action = main_actor.predict(state)
                action_noise = action + N #with N(Noise) #180개를 받아온다. 
                #Noise 조절이 좀 필요하다 step loop 안으로 들어와야 계속 변할수 있다. 
                action_noise = np.reshape(action_noise, (input_size_critic_a))
                
                #action_noise = np.transpose(action_noise) # [180, 1] <--이런식으로 받아오게 되므로 reshape해야함
                #print(action_noise.shape)
                
                
                ''' 
                print("action_noise")
                print(action_noise)
                print("type")
                print(type(action_noise))
                print(action_noise.shape)
                '''
                #여기는 actor가 잘 돌아가는지 확인하는 부분. tensorflow는 trainable 초기화는 항상 해주고 
                #placeholder에 X_input등을 넣어주면된다 (feed) 근데 여기서는 state가 input이니까 나오는게 맞음 feed안해도
                
                # Get new state and reward from environment  
                next_state, reward, done, record = environment.simulation(action_noise, st_step, record)
                
                #한 step의 reward씩 계속 reward_graph에 쌓는다. summation of reward
                reward_graph = reward + reward_graph
                
                if done:  
                    reward = -1

                ################ 이 부분이 Replay memory 부분이다 ##############
                buffer.append((state, action_noise, reward, next_state, done))
                if len(buffer) > buffer_memory:
                    buffer.popleft()
                
                #main을 target으로 복사한다 (critic)
                if state_step > 1 and state_step % target_update_frequency == 0:
                    sess.run(copy_ops_critic)
                    
                #main을 target으로 복사한다 (actor)
                if state_step > 1 and state_step % target_update_frequency == 0:
                    sess.run(copy_ops_actor)
  
                #print("update out")   
                ################################################################       
               
                state = next_state
                
                state_step = state_step + 1
                #print("step num : {}".format(step))
                
                if state_step == step_deadline:
                    break
       
            reward_graph = reward_graph/state_step
            
            #plt file로 reward graph 저장
            reward_record.write("%d %f \n" %(episode , reward_graph))
        
            state_step = 0
            #print("Episode : {} end ".format(episode))
            
           
            if episode % record_frequency == 0:
                _ = environment.record_end(record)
            # Episode (finish)
            
    reward_record.close()
            

if __name__ == "__main__":
    #여기가 main 맞으니까 main이 실행되는거 맞는데 굳이 위와 같이 할 필요가 있나 싶은데?
    main()
    
   
    print("All process is finished!")


