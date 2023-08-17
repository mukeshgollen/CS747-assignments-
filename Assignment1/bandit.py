import numpy as np
import argparse
import math


def epsilon_greedy_t1(bandits, seed, epsilon, horizon):
    np.random.seed(seed)
    num_bandits = len(bandits)
    curr_bandit = 0
    emp_mean = np.zeros(num_bandits, dtype=float)
    num_selection = np.zeros(num_bandits, dtype=int)
    rewards = 0
    regret = []
    most_optimal_bandit = max(bandits)
    
    for i in range(horizon):
        p = np.random.uniform(0, 1)
        
        if p<epsilon:
            curr_bandit = np.random.randint(0, num_bandits, dtype=int)
        else:
            curr_bandit = np.argmax(emp_mean)
    
        rewards = np.random.binomial(1, bandits[curr_bandit])
        emp_mean[curr_bandit] = (emp_mean[curr_bandit]*num_selection[curr_bandit] + rewards)/(num_selection[curr_bandit]+1)
        num_selection[curr_bandit] += 1
        regret.append(most_optimal_bandit - rewards)
        
    return np.sum(regret), 0    

def ucb(bandits, seed, scale, horizon):
    np.random.seed(seed)
    num_bandits = len(bandits)
    num_selection = np.zeros(num_bandits, dtype=int)
    sum_of_rewards = np.zeros(num_bandits, dtype=int)
    most_optimal_bandit = max(bandits)
    
    for i in range(horizon):
        curr_bandit = 0
        max_ub = 0
        
        for d in range(num_bandits):
            if(num_selection[d] > 0):
                mean_reward = sum_of_rewards[d]/num_selection[d]
                ucb_term = math.sqrt(scale * math.log(i+1)/num_selection[d])
                ub = mean_reward + ucb_term
            else: 
                ub = 1e200
            
            if ub > max_ub:
                max_ub = ub
                curr_bandit = d
            
        num_selection[curr_bandit] += 1
        reward =  np.random.binomial(1, bandits[curr_bandit])
        sum_of_rewards[curr_bandit] += reward

    return most_optimal_bandit*horizon - np.sum(sum_of_rewards), 0

def KL(x, y):
    if x == 1:
        return x*math.log(x/y)
    elif x == 0:
        return (1-x)*math.log((1-x)/(1-y))
    else:
        return x*math.log(x/y) + (1-x)*math.log((1-x)/(1-y))
    
def find(r, p):
    if p == 1:
        return 1

    l = p
    h = 1
    epsilon = 1e-5
    while (l+epsilon<h):
        x = (l+h)/2
        if(KL(p,x) <= r):
            l = x
        else:
            h = x   
    return l
            
def kl_ucb_t1_util(selection, bandit_rewards, bandit_i, num_bandits):
    kl_ucb_bandit = np.zeros(num_bandits, dtype=float)
    
    for i in range(num_bandits):
        p = bandit_rewards[i]/selection[i]
        kl_rhs = (math.log(bandit_i) + 3*math.log(np.log(bandit_i)))/selection[i]
        kl_ucb_bandit[i] = find(kl_rhs, p)
    return kl_ucb_bandit

def kl_ucb_t1(bandits, seed, horizon):
    np.random.seed(seed)
    
    most_optimal_bandit = max(bandits)
    num_bandits = len(bandits)
    rewards = np.zeros((num_bandits, horizon), dtype=int)
    
    for i in range(num_bandits):
        rewards[i, :] = np.random.binomial(1, bandits[i], horizon)
        
    total_rewards = 0
    curr_bandit = 0
    reward = 0
    selection = np.zeros(num_bandits, dtype=int)
    bandit_rewards = np.zeros(num_bandits, dtype=int)
    kl_ucb_bandits = np.zeros(num_bandits, dtype=float)
    
    for i in range(max(num_bandits, 3)):
        curr_bandit = i
        if curr_bandit >= num_bandits:
            curr_bandit = curr_bandit%num_bandits
            
        reward = rewards[curr_bandit, selection[curr_bandit]]
        selection[curr_bandit] += 1
        total_rewards += reward
        bandit_rewards[curr_bandit] += reward
        
    for i in range(max(num_bandits, 3), horizon):
        kl_ucb_bandits = kl_ucb_t1_util(selection, bandit_rewards, i, num_bandits)
        arg_kl_ucb = np.argmax(kl_ucb_bandits)
        
        curr_bandit = arg_kl_ucb
        reward = rewards[np.int64(curr_bandit), selection[np.int64(curr_bandit)]]
        selection[curr_bandit] += 1
        total_rewards += reward
        bandit_rewards[curr_bandit] += reward
    
    return most_optimal_bandit*horizon - total_rewards, 0           

def thompson_sampling_t1(bandits, seed, horizon):
    np.random.seed(seed)
    
    num_bandits = len(bandits)
    most_optimal_bandit = max(bandits)
    bandit_pulls = np.zeros(num_bandits, dtype=int)
    bandit_succ = np.zeros(num_bandits, dtype=int)
    bandit_fail = np.zeros(num_bandits, dtype=int)
    total_reward = 0
    
    for i in range(horizon):
        curr_bandit = 0
        prob_bandit = np.zeros(num_bandits, dtype=float)
        
        for x in range(num_bandits):
            prob_bandit[x] = np.random.beta(bandit_succ[x]+1, bandit_fail[x]+1)
        
        curr_bandit = np.argmax(prob_bandit)
        reward = np.random.binomial(1, bandits[curr_bandit])
        if reward == 1:
            bandit_succ[curr_bandit] += 1
        elif reward == 0:
            bandit_fail[curr_bandit] += 1
        
        bandit_pulls[curr_bandit] += (bandit_succ[curr_bandit] + bandit_fail[curr_bandit])
        total_reward += reward
        
    return most_optimal_bandit*horizon - total_reward, 0 
        

def alg_t3(mat, seed, horizon):
    np.random.seed(seed)
    
    num_reward = len(mat[0])
    num_bandits = len(mat)-1
    most_optimal_bandit = 0
    for i in range(num_bandits):
        temp = 0
        for j in range(num_reward):
            temp += mat[0][j]*mat[i+1][j]
        most_optimal_bandit = max(most_optimal_bandit, temp)
        
    bandit_pulls = np.zeros(num_bandits, dtype=int)
    bandit_succ = np.zeros(num_bandits, dtype=float)
    bandit_fail = np.zeros(num_bandits, dtype=float)
    total_reward = 0
    
    for i in range(horizon):
        curr_bandit = 0
        prob_bandit = np.zeros(num_bandits, dtype=float)
        
        for x in range(num_bandits):
            prob_bandit[x] = np.random.beta(bandit_succ[x]+1, bandit_fail[x]+1)
        
        curr_bandit = np.argmax(prob_bandit)
        reward = mat[0][int(np.random.choice(np.arange(0,num_reward), p=mat[curr_bandit+1]))]
        bandit_succ[curr_bandit] += reward
        bandit_fail[curr_bandit] += (1-reward)
        bandit_pulls[curr_bandit] += 1
        total_reward += reward
        
    return most_optimal_bandit*horizon - total_reward, 0

def alg_t4(mat, seed, threshold, horizon):
    np.random.seed(seed)
    
    num_reward = len(mat[0])
    num_bandits = len(mat)-1
    most_optimal_bandit = 0
    for i in range(num_bandits):
        temp = 0
        for j in range(num_reward):
            temp += mat[0][j]*mat[i+1][j]
        most_optimal_bandit = max(most_optimal_bandit, temp)
        
    bandit_pulls = np.zeros(num_bandits, dtype=int)
    bandit_succ = np.zeros(num_bandits, dtype=float)
    bandit_fail = np.zeros(num_bandits, dtype=float)
    total_reward = 0
    
    for i in range(horizon):
        curr_bandit = 0
        prob_bandit = np.zeros(num_bandits, dtype=float)
        
        for x in range(num_bandits):
            prob_bandit[x] = np.random.beta(bandit_succ[x]+1, bandit_fail[x]+1)
        
        curr_bandit = np.argmax(prob_bandit)
        reward = mat[0][int(np.random.choice(np.arange(0,num_reward), p=mat[curr_bandit+1]))]
        bandit_succ[curr_bandit] += reward
        bandit_fail[curr_bandit] += (1-reward)
        bandit_pulls[curr_bandit] += 1
        if reward >= threshold:
            total_reward += 1
        
    return 0, total_reward

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--instance', type=str ,help='Bandit instance file path', required=True)
    parser.add_argument('-al', '--algorithm', type=str ,help='Algorithm wants to run', required=True)
    parser.add_argument('-rs', '--randomSeed', type=int, help='random seed for randomization in algorithm', required=True)
    parser.add_argument('-ep', '--epsilon', type=float, help='epsilon for epsilon-greedy', default=0.02)
    parser.add_argument('-c', '--scale', type=float, help='scale value for task-2', default=2)
    parser.add_argument('-th', '--threshold', type=float, help='threshold for task-4', default=0)
    parser.add_argument('-hz', '--horizon', type=int, help='how many time algo needed to run', required=True)
    
    args = vars(parser.parse_args())
    
    instance_path = args['instance']
    algorithm = args['algorithm']
    seed = args['randomSeed']
    epsilon = args['epsilon']
    scale = args['scale']
    threshold = args['threshold']
    horizon = args['horizon']
    
    possible_algo = ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']
    
    # Double check the arguments
    # if (algorithm not in possible_algo):
    #     print("Please check your algorithm name")
    #     exit()
    
    # elif (seed < 0):
    #     print("Please check the random Seed limit")
    #     exit()
        
    # elif (epsilon > 1 or epsilon < 0):
    #     print("Please check the epsilon limit")
    #     exit()
        
    # elif (scale < 0):
    #     print("Please check the scale value")
    #     exit()
        
    # elif (threshold < 0  or threshold > 1):
    #     print("Please check the threshold limit")
    #     exit()
        
    # elif (horizon < 0):
    #     print("please give a positive horizon to run")
    #     exit()
        
    # Read bandit instance

    reg = 0
    highs = 0
    
    if(algorithm == 'epsilon-greedy-t1'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
    
        lines = list(map(float, lines))
    
        reg, highs = epsilon_greedy_t1(lines, seed, epsilon, horizon)
    
    elif(algorithm == 'ucb-t1' or algorithm == 'ucb-t2'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
    
        lines = list(map(float, lines))
        
        reg, highs = ucb(lines, seed, scale, horizon)
        
    elif(algorithm == 'kl-ucb-t1'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
    
        lines = list(map(float, lines))

        reg, highs = kl_ucb_t1(lines, seed, horizon)
        
    elif(algorithm == 'thompson-sampling-t1'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
    
        lines = list(map(float, lines))
        
        reg, highs = thompson_sampling_t1(lines, seed, horizon)
    
    elif(algorithm == 'alg-t3'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        
        mat = []
        for i in range(len(lines)):
            temp = list(map(float,lines[i].split()))
            mat.append(temp)
        
        reg, highs = alg_t3(mat, seed, horizon)
        
    elif(algorithm == 'alg-t4'):
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        
        mat = []
        for i in range(len(lines)):
            temp = list(map(float,lines[i].split()))
            mat.append(temp)
        
        reg, highs = alg_t4(mat, seed, threshold, horizon)
    
    print("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(instance_path, algorithm, seed, epsilon, scale, threshold, horizon, reg, highs))
    