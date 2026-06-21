import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from environment import easy21Env
from montecarlo import epsilon_greedy, monte_carlo, plot_value_function


def Sarsa_lambda(env, policy, llambda, num_episodes=100000):
    N_s = defaultdict(int)
    N_sa = defaultdict(int)
    Q = defaultdict(float)
    for episode in range(num_episodes):
        E = defaultdict(float) # eligibility traces
        state = env.reset()
        action = policy(env, Q, state)
        while True:
            E[(state, action)] += 1 # Eligibility trace incremented per visit(frequency)
            N_s[state] += 1
            N_sa[(state, action)] += 1
            next_state, reward, done = env.step(action)
            next_action = policy(env, Q, next_state) 
            if done:
                delta = reward - Q[(state, action)]
            else:
                delta = reward + Q[(next_state, next_action)] - Q[(state, action)]
            for sa in E: # update all states weve seen aka all states w/ an eligibility trace
                alpha = 1/N_sa[sa]
                Q[sa] += alpha * delta * E[sa]
                E[sa] *= llambda # decay by lambda (recency)
            if done:
                break
            state = next_state
            action = next_action
    return Q


# calculate mean squared error
def MSE(Q, Q_star, average=False):
    N = 0
    error = 0
    for dealer in range(1, 11):
        for player in range(1, 22):
            for action in env.actions:
                sa = ((dealer, player), action)
                error += ((Q[sa]) ** 2) - ((Q_star[sa]) ** 2)
                N += 1
    return error / N if average else error

def plot_MSE(env, policy, sarsa_lambda, Q_star, num_episodes=100000, average=True):
    lambdas = [i / 10 for i in range(11)]
    mses = []

    for llambda in lambdas:
        mse_runs = []
        print(f"Running SARSA (lambda={llambda})")
        for _ in range(10):
            Q = sarsa_lambda(env, policy, llambda, num_episodes=num_episodes)
            mse_runs.append(MSE(Q, Q_star, average=average))

        mses.append(np.mean(mse_runs))
    plt.figure()
    plt.plot(lambdas, mses, marker="o")
    plt.xlabel("Lambda")

    if average:
        plt.ylabel("Average MSE")
        plt.title("Average MSE vs Lambda")
    else:
        plt.ylabel("Summed Squared Error")
        plt.title("Summed Squared Error vs Lambda")
    
    plt.xticks(lambdas)
    plt.grid(True)
    plt.show()

    return lambdas, mses


env = easy21Env()
Q_star = monte_carlo(env, epsilon_greedy)
plot_MSE(env, epsilon_greedy, Sarsa_lambda, Q_star, average=True)