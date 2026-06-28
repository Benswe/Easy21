from environment import easy21Env
import numpy as np
import matplotlib.pyplot as plt
from montecarlo import monte_carlo, epsilon_greedy_mc
import random

DEALER_INTERVALS = [(1,4), (4, 7), (7, 10)]
PLAYER_INTERVALS = [(1,6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

def phi(env, state, action): # build and return the feature vector 
    features = np.zeros(36) # feature vector 
    dealer_card, player_sum = state
    action_index = env.actions.index(action)
    index = 0
    for d_low, d_high in DEALER_INTERVALS:
        for p_low, p_high in PLAYER_INTERVALS:
            for a_index in range(len(env.actions)):
                
                # check if feature at index i should be activated

                dealer_active = (d_low <= dealer_card <= d_high)
                player_active = (p_low <= player_sum <= p_high)
                action_active = (a_index == action_index)
                if dealer_active and player_active and action_active:
                    features[index] = 1
                index += 1
    return features
            


def Q_hat(env, theta, state, action): # theta*phi(state, action)
    return np.dot(phi(env, state, action), theta)

def epsilon_greedy(env, theta, state, epsilon=0.05): # use Q_hat to do this 
    if np.random.rand() < epsilon: # random case
        return(random.choice(env.actions))

    return max(env.actions, key=lambda action: Q_hat(env, theta, state, action))
    

# compare the optimal action-value function with our approximated av function
def compute_mse_approx(env, theta, q_star):
    error = 0
    N = 0
    for dealer in range(1, 11):
        for player_sum in range(1, 11):
            state = (dealer, player_sum)

            for action in env.actions:
                q_est = Q_hat(env, theta, state, action)
                q_true = q_star.get((state, action), 0.0)

                error += (q_est - q_true) ** 2
                N += 1

    return error/N # average mse

# uses Q_hat and eligiblity traces to update the action-value function
def sarsa_lambda(env, phi, policy, llambda, alpha=0.01, num_episodes=200000): 
    theta = np.zeros(36) # weights vector (what we update)

    for episode in range(num_episodes):
        eligibility = np.zeros(36)

        state = env.reset()
        action = policy(env, theta, state)
        while True:
            next_state, reward, done = env.step(action)
            # use policy to get next action
            next_action = policy(env, theta, state)
            if done:
                # target becomes reward when next state is terminal
                delta = reward - Q_hat(env, theta, state, action)
                # update eligibility 
                eligibility = llambda * eligibility + phi(env, state, action)
                theta += alpha*delta * eligibility # update the weights vector
                break
            # target - current_estimate
            delta = (reward + Q_hat(env, theta, next_state, next_action) - 
                    Q_hat(env, theta, state, action))
            eligibility = llambda * eligibility + phi(env, state, action)
            theta += alpha * delta * eligibility # update weights vector

            state = next_state
            action = next_action
    return theta
    

            




env = easy21Env()

q_star = monte_carlo(env, epsilon_greedy_mc)
theta = sarsa_lambda(env, phi, epsilon_greedy, 0.5)
print(compute_mse_approx(env, theta, q_star))