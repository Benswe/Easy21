import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from environment import easy21Env
from montecarlo import epsilon_greedy_mc, monte_carlo, plot_value_function


def sarsa_lambda(env, policy, llambda, num_episodes=500000):
    N_s = defaultdict(int)
    N_sa = defaultdict(int)
    Q = defaultdict(float)
    for episode in range(num_episodes):
        E = defaultdict(float) # eligibility traces
        state = env.reset()
        action = policy(env, Q, state, N_s)
        while True:
            E[(state, action)] += 1 # Eligibility trace incremented per visit(frequency)
            N_s[state] += 1
            N_sa[(state, action)] += 1
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - Q[(state, action)]
            else:
                next_action = policy(env, Q, next_state, N_s)
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
def MSE(env, Q, Q_star, average=False):
    N = 0
    error = 0
    for dealer in range(1, 11):
        for player in range(1, 22):
            for action in env.actions:
                sa = ((dealer, player), action)
                error += ((Q[sa]) - (Q_star[sa])) ** 2
                N += 1
    return error / N if average else error

def plot_MSE(env, policy, sarsa_lambda, Q_star, num_episodes=500000, average=True):
    lambdas = [i / 10 for i in range(11)]
    mses = []

    for llambda in lambdas:
        mse_runs = []
        print(f"Running SARSA (lambda={llambda})")
        for _ in range(10): # run over 10 episodes 
            Q = sarsa_lambda(env, policy, llambda, num_episodes=num_episodes)
            mse_runs.append(MSE(env, Q, Q_star, average=average))

        mses.append(np.mean(mse_runs)) # take mean of all runs for each lambda 
        
    
    plt.figure()
    plt.plot(lambdas, mses, marker="o") # lambdas vs mses 
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

def evaluate_policy(env, Q, num_games=100000):
    wins = 0
    draws = 0
    losses = 0
    
    for game in range(num_games):
        state = env.reset()
        while True:
            action = max(env.actions, key=lambda a: Q[(state, a)])
            next_state, reward, done = env.step(action)
            if done:
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                else:
                    draws += 1
                break
            state = next_state
    return wins/num_games, draws/num_games, losses/num_games


def policy_heatmap(Q): 
    for player in reversed(range(1, 22)):
        print(player, end="    ")
        for dealer in range(1, 11):
            if Q[(dealer, player), "hit"] > Q[(dealer, player), "stick"]:
                print("H", end=" ")
            else:
                print("S", end=" ")
        print()


if __name__ == "__main__":
    env = easy21Env()
    Q = sarsa_lambda(env, epsilon_greedy_mc, 0.1)
    q_star = monte_carlo(env, epsilon_greedy_mc)
    plot_MSE(env, epsilon_greedy_mc, sarsa_lambda, q_star)

    
    
