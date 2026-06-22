import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from environment import easy21Env

N0 = 100
N_sa = defaultdict(int)
N_s = defaultdict(int)


# this will be the policy we use to explore and gain experience to train on 
def epsilon_greedy(env, Q, state):

    epsilon = N0 / (N0 + N_s[state])
    random_num = random.random()
    if random_num > epsilon: # 1 - epsilon
        return(max(env.actions, key=lambda a: Q[(state, a)]))
    else:
        return(random.choice(env.actions))



# collect state action rewards for each time step
def trajectory(Q, env, policy):
    state = env.reset()
    trajectory = []
    while True:
        action = policy(env, Q, state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        if done:
            break
        state = next_state
    return trajectory


def monte_carlo(env, policy, num_episodes=750000):
    Q = defaultdict(float)
    for episode in range(num_episodes):
        G = 0
        # go in reverse in case we want to account for a discount factor later
        # get final reward first
        for state, action, reward in reversed(trajectory(Q, env, policy)): 
            G += reward
            N_sa[(state, action)] += 1
            N_s[state] += 1
            Q[(state, action)] += (1/N_sa[(state,action)]) * (G - Q[(state, action)]) # MC update
    return Q




env = easy21Env()

def plot_value_function(Q, env):
    dealer_values = np.arange(1, 11)
    player_values = np.arange(1, 22)

    V = np.zeros((21, 10))
    X, Y = np.meshgrid(dealer_values, player_values)
    for i, player_sum in enumerate(player_values):
        for j, dealer_card in enumerate(dealer_values):
            state = (dealer_card, player_sum)

            V[i, j]= max (
                Q.get((state, action), 0.0) for action in env.actions
            )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, V)

    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("V*(s)")
    ax.set_title("Easy21 Estimated Optimal Value Function")

    plt.show()

if __name__ == "__main__":
    Q = monte_carlo(env, epsilon_greedy) # Build Q

    plot_value_function(Q, env)