#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow.compat.v1 as tf
import random
from collections import deque
import dqn
import Environment as En
#from typing import List

environment = En.env() #call environment
#환경을 부른다. (사실 Environment라는 객체를 만든다)

_ = environment.reset() #환경을 초기화한다.

alpha = 0.9 #learning rate (based on Q)

#Input & Output
input_size = environment.state_num() #앞에서 환경을 초기화하고 환경 내에서는 state가 설정되고 input_size가 결정된다.
output_size = environment.action_setting() #환경에서의 action들을 모두 정의한다.


#Reinforcement learning parmeter
dis = 0.99  
buffer_memory = 50000 #Replay memory에 몇개를 넣을 것인가? (Buffer)
batch_size = 100 #Mini batch size Buffer에서 몇개씩 batch로 만들어서 학습시킬 것인가?


def replay_train(mainDQN_instance,targetDQN_instance, train_batch): #mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list:
   # 학습시킬 Network와 데이터 batch가 배달옴
    Q_old = np.empty(0)
    Q_new = np.empty(0)
    
    x_stack = np.empty(0)
    y_stack = np.empty(0)
    
    x_stack = np.reshape(x_stack, (0, mainDQN_instance.input_size))
    y_stack = np.reshape(y_stack, (0, mainDQN_instance.output_size)) 
    #print("train_Batch")
    #print(train_batch)
    for state, action, reward, next_state, done in train_batch: #이 부분들 다시 한 번 보도록 하자 (3번째 Cell에 연습함)

        Q = mainDQN_instance.predict(state)
        Q_old = np.max(Q)
        #DQN class module 만들어 놓은 거
        #predict하면 각 action에따른 Q값이 나와야 하는거 아닌가?
        
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = Q_old + alpha*(reward + dis*np.max(targetDQN_instance.predict(next_state)) - Q_old)
            Q_new = Q[0, action]
            #Error부분여기서 next_state가 next.state로 되어 있었음
            
        y_stack = np.vstack([y_stack, Q])
        #ValueError: all the input array dimensions for the concatenation axis must match exactly, 
        #but along dimension 1, the array at index 0 has size 4 and the array at index 1 has size 2 <위에 생긴 문제>
        x_stack = np.vstack([x_stack, state]) #state를 학습시키는거지 Q를 학습시키는건 아니다.
        
        #print(mainDQN_instance.update(x_stack, y_stack)) 출력해보면 그냥 하나의 array로 받아오는데,
        loss, _ = mainDQN_instance.update(x_stack, y_stack)
        
    return loss, Q_old, Q_new
    #return이 dqn.py module에서 loss를 결국 받아오도록 해놓았다.
    #mainDQN_instance.update(x_stack, y_stack) --> self.session.run([self._loss, self._train], feed), loss와 train을 받아옴.
        
        


def copy_var_ops(*, dest_scope_name ="target", src_scope_name = "main"):

    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    
    for src_var, dest_var in zip(src_vars, dest_vars): #zip의 기능은?
        op_holder.append(dest_var.assign(src_var.value()))
        #dest_var(tensor). assign
        
    return op_holder



def main():
    
    Q_old = np.empty(0)
    Q_new = np.empty(0)  
    st_step = 1 #action을 몇 time-step마다 취할 것인지에 대한 숫자
    state_step = 0
    record_frequency = 50
    step_deadline = 3000
    main_update_freq = 10
    target_update_frequency = 40 
    #main이 target을 향해서 update되어가고 이후에 target_update가 이루어져야 하기때문에 main_freq < target_update가 되어야 한다.
    max_episodes = 500
    
    # Replay buffer를 deque로 짠다. 
    buffer = deque() 
    #Memory는 50000개까지 

    reward_buffer = deque() #maxlen=100
    #reward_buffer또한 deque로 만들어서 마지막 100개까지 기억하도록 한다
    
    reward_record = open("reward.plt" , 'w', encoding='utf-8', newline='') 
    reward_record.write('VARIABLES = "Episode", "Reward" \n') 
    #Reward를 기록하기 위함.
    
    with tf.Session() as sess:
        
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main") #Class로 mainDQN 하나 만들고
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target") #Class로 targetDQN 하나 만들고
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net (copy하는 거) <처음에 targetDQN, mainDQN 임의로 설정이 되니까 같게 해줘야 한다>
        copy_ops = copy_var_ops(dest_scope_name="target",src_scope_name="main")
        #copy_var_ops는 위에 def 함수로 되어 있다.
        sess.run(copy_ops)


        for episode in range(0, max_episodes+1):
            
            print("Episode : {} start ".format(episode))

            e = 1.0 / ((episode / 10) + 1)
            done = False
            state = environment.reset() #envrionment로부터 state를 가져온다. (초기 state)
            reward_graph = 0
            
            
            ############### 두개의 Neural network로 학습을 시키는 부분이다 ##########
               
            #정확히는 Episode 10이 끝난 시점에서 update하는 것이다.
            if episode > main_update_freq and episode % main_update_freq == 0: # train every 10 episodes
            #if len(buffer)>10 and len(buffer) % batch_size*5 == 0:
                print("update start") #check!
                for _ in range(50):
                # Minibatch works better
                    #print("random_sample, step :{}" ,format(_)) #check complete
                    minibatch = random.sample(buffer, batch_size) 
                    minibatch = list(minibatch)
                    #print(minibatch.shape) #check!
                    #buffer에서 batch_size개수만큼씩 random하게 빼서 minibatch를 만든다.
                    #print("go to replay_train") #check complete
                    loss, Q_old, Q_new= replay_train(mainDQN, targetDQN, minibatch)
                
                    #if episode == 1 :
                    #print(minibatch)
                        
                    #Q_diff = abs(Q_old - Q_new)
                    #print("")
                    #print(Q_diff, Q_old, Q_new) 
            ########################################################################
            
            if episode % record_frequency == 0:
                record = environment.record_start(episode)
                
            while not done:
                
                if np.random.rand(1) < e:
                    action = environment.random_action()
                else:
                    action = np.argmax(mainDQN.predict(state)) 

                # Get new state and reward from environment
                
                next_state, reward, done, record = environment.simulation(action, st_step, record)
                
                #한 step의 reward씩 계속 reward_graph에 쌓는다. summation of reward
                reward_graph = reward + reward_graph
                
                if done:  
                    reward = -10

                ################ 이 부분이 Replay memory 부분이다 ##############
                buffer.append((state, action, reward, next_state, done))
                if len(buffer) > buffer_memory:
                    buffer.popleft()
                # buffer memory가 buffer_memory 이상이 되면 옛날 데이터는 빼버린다.
                #print("buffer") #check
                #print(type(buffer))
                
                    
                if state_step > 1 and state_step % target_update_frequency == 0:
                    #print("update target") #check complete
                    sess.run(copy_ops)
  
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


    
    
    
#First Error   
# File "C:\Users\thlee\DQN (real)\dqn.py", line 34
# W2 = tf.get_variable("W2", shape = [h_size, self.h_size], initializer = tf.contrib.layers.xavier_initializer())
#  ^ <위의 layer1 = tf.nn.tanh(tf.matmul(self._X, W1) <- 괄호 하나 빠져 있었음
# SyntaxError: invalid syntax

#Second ERROR
#ValueError: Sample larger than population or is negative
#뽑아낼게 전체 모집단보다 작으니까 뽑을게 없다는거잖아 아래 BATCH_SIZE를 64로 해서 그럼 이걸 줄여야함.
#batch_size = 64 #Mini batch size Buffer에서 몇개씩 batch로 만들어서 학습시킬 것인가?
#batch_size = 10으로 해보자.

#Third Error
# 위처럼 나타내면 결국 list - list로 되는 꼴인데
#이때 이런 TypeError: unsupported operand type(s) for -: 'list' and 'list'가 나온다
# 그 이유는 list는 -나 /에 정의되어 있지 않기때문이다.

#Fourth Error
#ValueError: not enough values to unpack (expected 4, got 3)    
#argv variable contains command line arguments. 
#In your code you expected 4 arguments, but got only 3 (first argument always script name). 

   


# In[ ]:


#namespace에 대한 확인용
def main():
    
    A = [3,5,6]
    
    print(A)
    
    
if __name__ == "__main__":
    main()


# In[25]:


# deque에 대한 연습용
from collections import deque

A = deque()

A.append(3)
A.append(5)
A.append(10)
A.append([5,8,10])

print(A)

A.popleft()

print(A)

A.pop()

print(A)


# In[65]:


#Random sample에 대한 확인용
B=np.empty(0)
print(B)
 
A=np.array([1, 3, 5])
B=A

print(B)

#np.empty는 그냥 비어 있는 공간을 만들게 된다.
#따라서 무엇이든 채울수 있다.

C=np.empty(0).reshape(0, 1)
#? 뭐지 왜 reshape한거지 그냥 하면 되는데 

print(C)

C=A
print(C)


#Random sample

J = [[3,4,5], [1,2,3], [3,5,5]]
print(J)
J_1 = random.sample(J, 1)
print(J_1)
J_2 = random.sample(J, 2)
print(J_2)
#Ok real random하게 뽑히네.


# In[72]:


#for문에 대한 확인용
import numpy as np 

# 전체 List를 마음대로 뽑을 수 있나를 확인해 보도록 한다 

A= np.array(([1,2,3], [3,4,5], [2,7,5], [1,3,6], [3,6,8])) #np.array 2차원 배열 (()) #두개를 넣어줘야 한다.

#print(A)

#zip함수에 대해서도 알아보도록 한다

for go, gi, zip_1 in A:
    print(go)
    print(gi)
    print(zip_1)
    print(go, gi, zip_1)
   


# In[3]:


#list array 확인
import numpy as np 

A = np.array([1,2,3])
B =list([1,2,3])
print(A, B)

print(A[0])
print(B[0])

