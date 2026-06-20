import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from environment import easy21Env
from montecarlo import epsilon_greedy, monte_carlo, plot_value_function

N_s = defaultdict(int)
N_sa = defaultdict(int)

def Sarsa_lambda(env, policy, llambda, num_episodes=100000):
    Q = defaultdict(float)
    E = defaultdict(float) # eligibility traces
    for episode in range(num_episodes):
        state = env.reset()
        action = policy(env, Q, state)
        while True:
            E[(state, action)] += 1 # Eligibility trace incremented per visit(frequency)
            N_s[state] += 1
            N_sa[(state, action)] += 1
            alpha = 1/N_sa[(state, action)]
            next_state, reward, done = env.step(action)
            next_action = policy(env, Q, state) 
            if done:
                delta = reward - Q[(state, action)]
            else:
                delta = reward + Q[(next_state, next_action)] - Q[(state, action)]
            for sa in E: # update all states weve seen aka all states w/ an eligibility trace
                Q[(state, action)] += alpha * delta * E[sa]
                E[sa] *= llambda # decay by lambda (recency)
            if done:
                break
            state = next_state
            action = next_action
    return Q

env = easy21Env()

Q = Sarsa_lambda(env, epsilon_greedy, 0) 

Q_star = monte_carlo(env, epsilon_greedy)



# calculate mean squared error
def MSE(env, Q, Q_star):
    error = 0
    for sa in Q:
        error += (Q[sa] - Q_star[sa]) ** 2
    return error

error = MSE(env, Q, Q_star)
print(error)