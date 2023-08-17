'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        np.random.seed(0)
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.1
        self.epsilon_T2 = 0.02
        self.learning_rate_T1 = 0.1
        self.learning_rate_T2 = 0.325
        self.num_position = 25
        self.num_velocity = 25
        self.num_tilings = 4
        self.num_action = 3
        self.weights_T1 = np.random.rand(self.num_position, self.num_velocity, self.num_action)
        self.weights_T2 = np.random.rand(self.num_position, self.num_velocity, self.num_action, self.num_tilings)
        self.discount = 1
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        position_space = np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.num_position)
        velocity_space = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.num_velocity)
        position, velocity = obs
        position_bin = np.digitize(position, position_space)
        velocity_bin = np.digitize(velocity, velocity_space)

        return [position_bin,velocity_bin]
        
    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):
        position, velocity = obs
        state_list = []
        for i in range(self.num_tilings):
            position_space = np.linspace(-1.2 - 0.054 + i*0.0018, 0.6 + i*0.0018, 25)
            velocity_space = np.linspace(-0.07 -0.042 + i*0.0014, 0.07 + i*0.0014, 25)
            position_bin = np.digitize(position, position_space)-1
            velocity_bin = np.digitize(velocity, velocity_space)-1
            state_list.append([position_bin, velocity_bin])
        return state_list

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_action)
        else:
            if len(weights.shape)==3:
                return np.argmax(weights[state[0]][state[1]])
            else:
                weights_list = np.zeros(self.num_action)
                for a in range(self.num_action):
                    for i in range(self.num_tilings):
                        weights_list[a] += weights[state[i][0]][state[i][1]][a][i]
                return np.argmax(weights_list)

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        if len(weights.shape)==3:
            weights[state[0]][state[1]][action] += learning_rate * (reward + self.discount * weights[new_state[0]][new_state[1]][new_action] - weights[state[0]][state[1]][action])
            self.weights_T1 = weights
        else:
            for i in range(self.num_tilings):
                weights[state[i][0]][state[i][1]][action][i] += learning_rate * (reward + self.discount * weights[new_state[i][0]][new_state[i][1]][new_action][i] - weights[state[i][0]][state[i][1]][action][i])
            self.weights_T2 = weights
        return weights
    

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    # if t<190:
                        # print(t)
                    reward_list.append(-t)
                    break
                t += 1
            if e % 100 == 0:
                print(f"{e} episode done!")
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
    - load_data: Ungraded.
    - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
    - save_data: Ungraded.
    - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    task='T2'
    # train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    agent.train(task)
    print(agent.test(task))
