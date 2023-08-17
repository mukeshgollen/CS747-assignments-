import argparse
import numpy as np

def decoder(policy_addr, state_addr, player):
    print(player)
    with open(policy_addr, 'r') as f:
        policy_reader = f.readlines()
        policy_reader = [p.strip() for p in policy_reader]
        policy = []
        
        for p in policy_reader:
            p = p.split()
            policy.append(int(float(p[1])))
        
        with open(state_addr, 'r') as f1:
            state_reader = f1.readlines()
            numstates = len(state_reader)
            
            states = [s.strip() for s in state_reader]
            
            temp = " 0 0 0 0 0 0 0 0 0"
            for i in range(numstates):
                p = 2*(policy[i]+1)-1
                curr = temp[:p] + '1' + temp[p+1:]
                print(states[i], curr)
                

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-policy", type=str)
    parser.add_argument("--states", type=str)
    parser.add_argument("--player-id", type=int)
    args = vars(parser.parse_args())
    policy_addr = args['value_policy']
    state_addr = args['states']
    player = args['player_id']
    
    decoder(policy_addr, state_addr, player)