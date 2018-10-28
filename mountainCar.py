import random
import gym
import numpy as np
from tensorflow import keras
from collections import deque
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

ENV_NAME = "MountainCar-v0"


class NNQ:
   
   
    def __init__(self,input_space,action_space):
        
        #HyperParameters
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.002
        self.MEMORY_SIZE = 1000000
        self.BATCH_SIZE = 30
        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.997
        self.exploration_rate = self.EXPLORATION_MAX
        self.reward = 0
        
        self.actions = action_space
        #Experience Replay 
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        
        #Create the NN model
        self.model = Sequential()
        self.model.add(Dense(64,input_shape =(input_space,),activation="relu"))
        self.model.add(Dense(64,activation="relu"))
        self.model.add(Dense(self.actions,activation="softmax"))
        self.model.compile(loss="mse",optimizer=Adam(lr = self.LEARNING_RATE))
        
        
    def act(self,state):
        #Exploration vs Exploitation
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.actions)
        
        q_values  = self.model.predict(state)
        
        return np.argmax(q_values[0])
    
    def remember(self,state,action,reward,next_state,done):
        #in every action put in the memory
        self.memory.append((state,action,reward,next_state,done))
    
    
    def experience_replay(self):
        #When the memory is filled up take a batch and train the network
        if len(self.memory) < self.MEMORY_SIZE:
            return
        
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state,action,reward,next_state, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA*np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state,q_values,verbose=0)
        
        if self.exploration_rate > self.EXPLORATION_MIN:
            self.exploration_rate *= self.EXPLORATION_DECAY


def mountainCar():
    env = gym.make(ENV_NAME)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    maxPosition = -2

    nnq = NNQ(observation_space, action_space)
    run = 0
    flag = False
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        total_rewards = 0
        while True:
            step += 1
            if flag == True:
                env.render()
            
            
            action = nnq.act(state)
            state_next, reward, terminal, info = env.step(action)
            #If the car pulls back on the left or right hill he gets a reward of +20
            if state_next[1] > state[0][1] and state_next[1]>0 and state[0][1]>0:
                reward = 20
            elif state_next[1] < state[0][1] and state_next[1]<=0 and state[0][1]<=0:
                reward =20
            #if he finishes with less than 200 steps
            if terminal and step < 200:
                reward += 10000
            else:
                reward += -25
            total_rewards += reward
            state_next = np.reshape(state_next, [1, observation_space])
            nnq.remember(state, action, reward, state_next, terminal)
            nnq.experience_replay()
            state = state_next
            if terminal:
                if step < 200 :
                    flag = True
                    print("Successful Episode!")
                print ("Run: " + str(run) + ", exploration: " + str(nnq.exploration_rate) + ", score: " + str(step))
                break
           


if __name__ == "__main__":
    mountainCar()

