#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Bugers equation (two wave number)
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import csv

################## Global variables declare #################
#fixed variable (Placehold)
N = np.zeros([1],dtype = np.int32)
nu = np.zeros([1],dtype = np.float64)
Action_num = np.zeros([1],dtype = np.int32)
d_t = np.zeros([1],dtype = np.int32)
d_x = np.zeros([1],dtype = np.int32)
Domain = np.zeros([1],dtype = np.int32)


N = 180
nu = 0.1
Action_num = 400 #action의 개수를 촘촘히 할 수록 더 부드럽게 될듯
Domain = 2.0*np.pi
d_t = 1/200
d_x = Domain/N


#Variables can be changed
wave = np.zeros([N],dtype = np.float64)
x = np.zeros([N],dtype = np.float64)
u_old = np.zeros([N],dtype = np.float64)
U_old = np.zeros([N],dtype = np.complex128)
U_new = np.zeros([N],dtype = np.complex128)
G_old = np.zeros([N],dtype = np.complex128)
state = np.zeros([N],dtype = np.float64)
action = np.zeros([Action_num],dtype = np.float64)
force = np.zeros([N],dtype = np.float64)
sin_half = np.zeros([N],dtype = np.float64)
cos_1 = np.zeros([N],dtype = np.float64)
#############################################################


class env(): 

    def reset(self): #초기화하고 state를 보내준다
        
        ##################### Global varible for burgers equation #################
        global wave, x, u_old, U_old, U_new, G_old, state
        global N, d_x, d_t, Domain
        ############################################################################
        
        #####한 번 다시 0으로 초기화시켜주면서 선언해준다######
        wave = np.zeros([N],dtype = np.float64)
        x = np.zeros([N],dtype = np.float64)
        u_old = np.zeros([N],dtype = np.float64)
        U_old = np.zeros([N],dtype = np.complex128)
        U_new = np.zeros([N],dtype = np.complex128)
        G_old = np.zeros([N],dtype = np.complex128)
        state = np.zeros([N],dtype = np.complex128) 
        ########################################################
        
        #Initial condition (Discretization)
        for i in range (N):
            x[i] = i*d_x  

            if(i<N//2):
                wave[i] = i   ##wave number
        
            elif(i>=N//2):
                wave[i] = i-N

        for i in range (N):
            u_old[i] = np.sin(x[i]) 
            cos_1[i] = np.cos(x[i])
         
        state = u_old
        
        return state

     

    def state_num (self):
        
        state_num = state.shape[0]
        
        return state_num

    
    
    def action_setting (self): #action number - output_size를 return한다.
        
        global force, sin_half, Action_num, action, cos_1
  
        for i in range (Action_num):           
            
            if(i<Action_num//2):
                action[i] = ((i)/float(Action_num))/10   ##action num -1 to 1
        
            elif(i>=Action_num//2):
                action[i] = (-(Action_num - i)/float(Action_num))/10
            
           
        #print(action)       
        #Sin파형의 forcing
        for i in range(N):
            force[i] = 0.3*cos_1[i] #sin forcing
            cos_1[i] = 0.5*cos_1[i]
        
        
        #위에서는 action x^2에 곱할것을 따로 만들어줌
        
        '''
         #x^2파형의 forcing
        for i in range (N): #기본적으로 forcing을 주는 term
            force[i] = 0.001*(-(x[i]- np.pi)**2 + 3) 
            
            
            if(i<=N//4-10 or i>=N*3//4+10):
                force[i] = 0.0 #Zero-padding
                        
                #print(force[i])     
        '''

        if __name__ == "__main__":
            print("action list")
            print(action)
        
        return Action_num
    

    
    def random_action(self):
        
        global Action_num
        
        index = np.arange(Action_num)
        index = list(index)
        
        rand_action = random.sample(index, 1) 
   
        return rand_action
    
   
    
    
    def record_start(self, episode):
        
        global record
        
        record = open("Episode{}.plt" .format(episode), 'w', encoding='utf-8', newline='') 
        record.write('VARIABLES = "x", "y" \n')
        print("record start")
        
        return record
    
    
    
    
    
    def simulation(self, arg_action, T, record): 
        
        global u_old, U_old, wave, U_new, G_old, state, action, force, sin_half, cos_1
        global nu, d_t, N, f

        
        summation = np.zeros([1], dtype = np.float64)
        #위에서 받아온 action index를 주어서 실제 action 값을 받아오도록 한다.
        max_action = np.zeros([N], dtype = np.float64)
        
        #print(max_action.shape)
        for i in range (N):
            max_action[i] = action[arg_action]*force[i] #Action을 설정하는 부분.
        #action[arg_action]    
        
        
        if __name__ == "__main__":
            print("max")
            print(max_action)
        
        #u_old = state #현재 global variable로 u_old를 가져왔는데도 unbound error가 떠서 다시 state받아왔던걸 u_old로 옮겼다.
        

        for t in range(T):
            
            try: 
                record.write('Zone T= "%d"\n' %t)  
                record.write('I=%d \n' %N)
                
            except ValueError:
                None 
            
            
            #state가 next_state로 넘어가는기준 time-step 10번마다 한 번의 state가 넘어간다.
                
            w_old = u_old**2
            U_old = np.fft.fft(u_old)  
            G_old = np.fft.fft(w_old)
            Action = np.fft.fft(max_action)
            #print(G_old)

            #Initial condition (Discretization) $ Inverse Fourier
    
            #for i in range (N): # without zeropadding
            for i in range (N): # with zeropadding
        
                U_new[i] = (1-d_t/2*nu*(wave[i]**2))/(1+d_t/2*nu*(wave[i]**2))*U_old[i] - d_t*wave[i]/(1+d_t/2*nu*(wave[i]**2))*G_old[i]*1j
                #U_new[i] = U_new[i] + Action[i]
        
        
                if(i>=N//3 and i<N*2//3):
                    U_new[i] = 0.0 #Zero-padding
                
            
            u_old = np.real(np.fft.ifft(U_new)) #Inverse Fourier transform

            
            try: 
                for k in range(N):
                    record.write("%d %f \n" %(k , u_old[k]))
                    
            except ValueError:
                None
            #U (Frequency domain), u (real domain)             
        
        
        #Update action to environment        
        #if T>10 and T% 10 ==0:
        u_old = u_old + max_action #Do action!
        
        
        ########### Reward function ##########        
        diff = abs(u_old - cos_1)
        reward = -6*np.log(diff + 0.001)
        #착가했었던게 이 reward를 모든 Position에 대해서 받아버림. 모든 Position에 대해서 평균내야한다.
        
        for i in range (N):
            summation = reward[i] + summation
        
        avg_reward = summation/N
        #print(avg_reward)
        
        #np.log(x) = ln(x)
        ######################################
        
        #next_state를 넣어서 보내줘야하므로
        next_state = u_old
            
            
        #done (너무 이상하게 돼서 episode를 끝내버리는 경우)
        for _  in range (N):
            
            if u_old[i] >10:
                done = True
                
            else:
                done = False
       
        #done의 경우는 새로 나온 u값이 10이상인 수치가 있으면 Episode를 끝낸다.
        
 
      
        ''' #Plot the graph
        #plt.subplots(nrows=2, ncols=1) graph를 1개 이상을 보여주고 싶을때 사용한다
        plt.plot(x[0:180],u_old[0:180],label='u-velocity')
        #plt.plot(x[0:51],x[0:51],label='x')
        plt.legend()
        plt.xlabel('Distance'); plt.ylabel('u-velocity'); plt.grid()
           
        ''' 
        
        return next_state, avg_reward, done, record
    
    
    def record_end(self, record):
        
        record.close()
        print("record finish")
        
        return 0


                    
#RuntimeWarning: overflow encountered in square (Sin파형과 discrete한 0.3같은걸 따로 주면) 왼쪽과 같은 오류가 뜬다. (절벽이라 그럴듯)
# 이제 알았다. forcing 유량이라고 생각하면 decaying해가는것보다 더 빠르게 증가해서 결국 발산해버려서 값을 정의하지 못하게 되는것이다.
            
