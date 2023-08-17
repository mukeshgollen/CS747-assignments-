import numpy as np
import argparse
from collections import defaultdict

np.random.seed(10)

def checker(state, opponent):
    winner = 0
    num_zero = state.count('0')
    if (state[0:3] == '222' or state[3:6] == '222' or state[6:9] == '222' 
        or (state[0]=='2' and state[3]=='2' and state[6]=='2')
        or (state[1]=='2' and state[4]=='2' and state[7]=='2') 
        or (state[2]=='2' and state[5]=='2' and state[8]=='2')
        or (state[0]=='2' and state[4]=='2' and state[8]=='2')
        or (state[2]=='2' and state[4]=='2' and state[6]=='2')):
        winner = 1
    
    elif (state[0:3] == '111' or state[3:6] == '111' or state[6:9] == '111' 
        or (state[0]=='1' and state[3]=='1' and state[6]=='1')
        or (state[1]=='1' and state[4]=='1' and state[7]=='1') 
        or (state[2]=='1' and state[5]=='1' and state[8]=='1')
        or (state[0]=='1' and state[4]=='1' and state[8]=='1')
        or (state[2]=='1' and state[4]=='1' and state[6]=='1')):
        winner = 2
    
    if winner == 0:
        if num_zero == 0:
            return 'draw'
        else:
            return 'continue'
    elif winner == opponent:
        return 'lose'
    elif winner != opponent: 
        return 'win'

def encode(policy_addr, state_addr):
    numstates = 0
    opponent = 0
    
    with open(policy_addr, 'r') as f:
        policy_reader = f.readlines()
        policy_reader = [p.strip() for p in policy_reader]
        opponent = int(policy_reader[0])
        policy = {}
        policy = defaultdict(list)
        for i in range(1,len(policy_reader)):
            temp = policy_reader[i].split()
            policy[str(temp[0])] = temp[1:]
        
        with open(state_addr, 'r') as f1:
            state_reader = f1.readlines()
            numstates = len(state_reader)
            print(f"numStates {numstates+1}")
            print(f"numActions {9}")
            print(f"end {numstates}")
            
            states = [s.strip() for s in state_reader]
            
            if opponent == 2:
                for ii,s in enumerate(states):
                    for action in range(0,9):
                        
                        current_state = s
                        prev_state = s
                        prev_idx = ii
                            
                        # base case
                        if current_state[action] != '0':
                            print(f"transition {prev_idx} {action} {prev_idx} {-2.000000} {1.000000}")
                            continue
                            
                        current_state = current_state[:action]+'1'+current_state[action+1:]
                        condition = checker(current_state, 2)
                        
                        if condition == 'draw' or condition == 'lose':
                            print(f"transition {prev_idx} {action} {numstates} {0.000000} {1.000000}")
                            continue
                        
                        # search in policy of opponent
                        prob = policy[current_state]
                        
                        temp = current_state
                        for i, p2 in enumerate(prob):
                            p = float(p2)
                            # print(i, p)
                            if p > 0:
                                current_state = temp[:i] + '2' + temp[i+1:]
                                try:
                                    curr_idx = states.index(current_state)
                                except:
                                    curr_idx = numstates
                                
                                # take action
                                condition = checker(current_state, 2)
                                if condition == 'continue':
                                    print(f"transition {prev_idx} {action} {curr_idx} {0.000000} {p}")
                                    # continue
                                elif condition == 'draw' or condition == 'lose':
                                    if current_state not in states:
                                        print(f"transition {prev_idx} {action} {numstates} {0.000000} {p}")
                                    else:
                                        print(f"transition {prev_idx} {action} {curr_idx} {0.000000} {p}")
                                elif condition == 'win':
                                    print(f"transition {prev_idx} {action} {curr_idx} {1.000000} {p}")
                                    
            elif opponent == 1:
                for ii,s in enumerate(states):
                    for action in range(0,9):
                        
                        current_state = s
                        prev_state = s
                        prev_idx = ii
                        
                        # base case
                        if current_state[action] != '0':
                            print(f"transition {prev_idx} {action} {prev_idx} {-2.000000} {1.000000}")
                            continue
                            
                        current_state = current_state[:action]+'2'+current_state[action+1:]
                        
                        condition = checker(current_state, 1)
                        
                        if condition == 'draw' or condition == 'lose':
                            print(f"transition {prev_idx} {action} {numstates} {0.000000} {1.000000}")
                            continue
                        
                        # search in policy of opponent
                        prob = policy[current_state]
                        
                        temp = current_state
                        for i, p2 in enumerate(prob):
                            p = float(p2)
                            if p > 0:
                                current_state = temp[:i] + '2' + temp[i+1:]
                                try:
                                    curr_idx = states.index(current_state)
                                except:
                                    curr_idx = numstates
                                
                                # take action
                                condition = checker(current_state, 1)
                                if condition == 'continue':
                                    print(f"transition {prev_idx} {action} {curr_idx} {0.000000} {p}")
                                    
                                elif condition == 'draw' or condition == 'lose':
                                    if current_state not in states:
                                        print(f"transition {prev_idx} {action} {numstates} {0.000000} {p}")
                                    else:
                                        print(f"transition {prev_idx} {action} {curr_idx} {0.000000} {p}")
                                    
                                elif condition == 'win':
                                    print(f"transition {prev_idx} {action} {curr_idx} {1.000000} {p}")
    
    print("mdptype episodic")
    print("discount  1.0")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",type=str)
    parser.add_argument("--states",type=str)
    args = parser.parse_args()
    policy_addr = args.policy
    state_addr = args.states
    
    encode(policy_addr, state_addr)