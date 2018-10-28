# Using-DQN-on-Mountain-Car

In this project we implement a Deep Neural Network to solve the Mountain Car -v0  problem of Open AI Gym.


You can click the link below to read about the environment:
                  https://gym.openai.com/envs/MountainCar-v0/

The trick used to solve this problem was to modify the rewards that was given. The environment were given the rewards below:
  -1 in every step (total steps were 200).
  +1 if it managed to reach the goal.

In our case we modified these rewards:
  -25 at every step
  +20 if the car pulls back on the left hill or moves forward on the right hill
  +10000 if he reaches the goal

We used experience replay with memory of 1000000 size and when the memory is filled up the network is training with batches of 30 from the memory. From the experiments we saw that using a Neural Network with 2 hidden layers of 64 neurons with relu activation function, after the memory is filled up the agent can reach the goal after 2-5 trainings.
