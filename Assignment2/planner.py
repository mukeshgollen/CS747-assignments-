import argparse
import numpy as np
import pulp
from scipy.sparse import csr_matrix

class MDP:
    def __init__(self):
        self.numstates = 0.0
        self.numactions = 0.0
        self.type = 'continuing'
        self.discount = 0.0
        self.theta = 1e-9
        self.delta = np.float(0)
    
    def read_file(self, mdp_addr):
        
        with open(mdp_addr, 'r') as f:
            mdp_reader = f.readlines()
            
        for elem in mdp_reader:
            elem_list = elem.split()
            # print(elem_list)
            if elem_list[0] == 'numStates':
                self.numstates = int(elem_list[1])
            
            elif elem_list[0] == 'numActions':
                self.numactions = int(elem_list[1])
                
            elif elem_list[0] == 'end':
                self.end = elem_list[1:]
                self.transition = x = [[[] for i in range(self.numactions)] for j in range(self.numstates)]
                self.policy = np.zeros(int(self.numstates))
                self.valuefunc = np.zeros(int(self.numstates))
            
            elif elem_list[0] == 'mdptype':
                self.type = elem_list[1]
            
            elif elem_list[0] == 'discount':
                self.discount = np.float64(elem_list[1])
                            
            elif elem_list[0] == 'transition':
                self.transition[int(elem_list[1])][int(elem_list[2])].append([int(elem_list[3]), float(elem_list[4]), float(elem_list[5])])
            
        
    def find_value(self, state):
        act = np.zeros(self.numactions)
        
        for j in range(self.numactions):
            for i in range(self.numstates):
                for elem in self.transition[state][j]:
                    if len(elem) > 0 and elem[0] == i:
                        act[j] += (elem[2] * (elem[1] + self.discount*self.valuefunc[elem[0]]))    
        return act
    
    def find_value1(self, state, action):
        v = 0
        for s1, r, t in self.transition[state][action]:
            v += t * ( r + self.discount*self.valuefunc[s1])
        
        return v
    
    def valueiteration(self):
        
        while True:
            self.delta = 0
            for s in range(self.numstates):
                v = self.valuefunc[s]
                self.valuefunc[s] = max(self.find_value(s))
                self.delta = max(self.delta, np.abs(v - self.valuefunc[s]))
            if self.delta < self.theta:
                break
        
        for s in range(self.numstates):
            self.policy[s] = np.argmax(self.find_value(s))
        
        return self.policy, self.valuefunc
    
    def linearprogramming(self):
        lp = pulp.LpProblem("Linear_programming", pulp.LpMinimize)
        
        eq_var = []
        for s in range(self.numstates):
            v = pulp.LpVariable('X'+str(s))
            eq_var.append(v)
            
        lp += sum(eq_var)
        
        for s in range(int(self.numstates)):
            for a in range(int(self.numactions)):
                    lp += (eq_var[s] >= pulp.lpSum([t * (r + self.discount*eq_var[s1]) for s1, r, t in self.transition[s][a]]))
        
        rslt = lp.solve(pulp.PULP_CBC_CMD(msg=0))
        
        for v in lp.variables():
            self.valuefunc[int(v.name[1:])] = v.varValue
        
        t = self.type
        
        V0 = np.ones(self.numstates)
        while(np.linalg.norm(V0 - self.valuefunc) > self.theta):
            V0 = self.valuefunc
            for s in range(self.numstates):
                values = []
                for a in range(self.numactions):
                    value = 0
                    for s1, r, t in self.transition[s][a]:
                        value += t * ( r + self.discount * V0[s1])
                    values.append(value)
                self.valuefunc[s] = np.max(values)
                self.policy[s] = np.argmax(values)
        
        return self.policy, self.valuefunc
    
    def howardpolicy(self):
        change = True
        temp_policy = np.zeros(self.numstates)
        
        while change:    
            while True:
                self.delta = 0
                for s in range(self.numstates):
                    v = self.valuefunc[s]
                    self.valuefunc[s] = max(self.find_value(s))
                    self.delta = max(self.delta, abs(v - self.valuefunc[s]))
                
                if self.delta < self.theta:
                    break
            
            for s in range(self.numstates):
                temp_policy[s] = np.argmax(self.find_value(s))
            
            if (temp_policy == self.policy).all():
                change = False
            
            self.policy = temp_policy
        
        return self.policy, self.valuefunc

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mdp",type=str)
    parser.add_argument("--algorithm",type=str, default='lp')
    
    args = parser.parse_args()
    
    mdp_addr = args.mdp
    algo = args.algorithm
    
    mdp_obj = MDP()
    mdp_obj.read_file(mdp_addr)
    
    
    if algo == 'vi':
        policy, value = mdp_obj.valueiteration()
        
    elif algo == 'hpi':
        policy, value = mdp_obj.howardpolicy()
        
    elif algo == 'lp':
        policy, value = mdp_obj.linearprogramming()
    
    for p,v in zip(policy, value):
        print(v, " ", p)