import numpy as np
from os import system

state_file1 = './data/attt/states/states_file_p1.txt'
state_file2 = './data/attt/states/states_file_p2.txt'

numstates1 = 0
numstates2 = 0

with open(state_file1, 'r') as f:
    state_reader = f.readlines()
    numstates1 = len(state_reader)
    states1 = [s.strip() for s in state_reader]
    
with open(state_file2, 'r') as f:
    state_reader = f.readlines()
    numstates2 = len(state_reader)
    states2 = [s.strip() for s in state_reader]
    
with open(f"p{2}_policy{1}", 'w') as f:
    f.write('2\n')
    for i in range(numstates2):
        zero_count = states2[i].count('0')
        nums = np.random.random(zero_count)
        nums /= nums.sum()
        prob = ""
        x = 0
        for e in states2[i]:
            if e == '0':
                prob += (" "+str(nums[x]))
                x+=1
            else:
                prob += " 0.0"
        f.write(states2[i] + prob+'\n')

system(f"python3 encoder.py --policy ./p2_policy{1} --states {state_file1} > tempmdpfile")
system(f"python3 planner.py --mdp ./tempmdpfile > temp_value_policies")
system(f"python3 decoder.py --value-policy temp_value_policies --states {state_file1} --player-id 1 > p1_policy{1}")
        
print(f"iteration {1}")

it = 1
for ii in range(10):
    # take statefile2 and policy p1 and make policy p2
    system(f"python3 encoder.py --policy ./p1_policy{it} --states {state_file2} > tempmdpfile")
    system(f"python3 planner.py --mdp ./tempmdpfile > temp_value_policies")
    system(f"python3 decoder.py --value-policy temp_value_policies --states {state_file2} --player-id 2 > p2_policy{it+1}")
    
    
    # take statefile1 and policy p2 and make policy p1
    system(f"python3 encoder.py --policy ./p2_policy{it+1} --states {state_file1} > tempmdpfile")
    system(f"python3 planner.py --mdp ./tempmdpfile > temp_value_policies")
    system(f"python3 decoder.py --value-policy temp_value_policies --states {state_file1} --player-id 2 > p1_policy{it+1}")
    
    
    print(f"iteration {it+1}")
    print("Difference for Policy 1")    
    system(f"diff -y --suppress-common-lines p1_policy{it} p1_policy{it+1} | wc -l")
    print("Difference for Policy 2")    
    system(f"diff -y --suppress-common-lines p2_policy{it} p2_policy{it+1} | wc -l")
    
    it+=1
