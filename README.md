# EASY 21

## Overview
Easy21 is a game very similar to blackjack, but with a twist. There are no aces or face cards, and the color of the card matters. If a player decides to hit and receives a black card, his total increases based on the value of the card, just like in regular blackjack. However, if he receives a red card his total is subtracted by the value of the card. The dealer has the same rules, however, she has an advantage in that she only decides what to do after the player has decided to stick. Thus the player can bust and the dealer never has to play, or the player sticks and the dealer knows exactly what they must hit. The dealer plays a default strategy of sticking on a sum of 17+, and hitting otherwise. This strategy proved to be valid as it won ≈ 63% of hands over a random player strategy. Thus, this sets up an interesting RL problem in which a policy that would be difficult for a human to derive must be found. 

## Algorithms Implemented

### Monte-Carlo Control
I first implemented Monte-Carlo control to estimate the optimal action-value function, (Q(s, a)). Monte-Carlo methods learn from complete episodes, meaning the value of each state-action pair is updated only after the game has finished and the final return is known.

For each episode, the agent follows an epsilon-greedy policy. This allows it to mostly choose the action it currently believes is best, while still exploring other actions often enough to discover better strategies. After the episode ends, the return is used to update the value estimate for each visited state-action pair.

Over many episodes, the action-value function improves, and the policy becomes stronger. I used the final Monte-Carlo estimate as an approximation of (Q^*), which then served as a baseline for comparing the performance of SARSA(λ) and function approximation.


### Sarsa(λ)
I then implemented Sarsa(λ), this is a more interesting algorithm that seeks to improve upon Monte-Carlo by allowing for on-policy updates. What this means is that after every single time-step, you are able to update your value function for that state/action pair. What makes Sarsa(λ) even nicer is that it at every time-step it doesn't just update the state/action pair it was just in, it updates every state-action pair seen in that episode based on how eligible it is to receive part of the current TD error. 

It does this using eligibility traces which assign credit to each state based on frequency and recency. When a TD error is observed, that error is distributed backward through the trace, allowing earlier decisions in the episode to be updated as well. The λ parameter controls how far credit is assigned backward. When λ is close to 0, Sarsa(λ) behaves more like one-step TD learning. When λ is close to 1, it behaves more like Monte-Carlo learning, where updates are influenced by long-term returns.

### Linear Function Approximation

I also implemented SARSA(λ) with linear function approximation using a coarse-coded feature representation. The approximate action-value function is represented as:

$$
\hat{Q}(s, a; \theta) = \theta^\top \phi(s, a)
$$

where $\theta$ is the learned weight vector and $\phi(s, a)$ is the feature vector for a given state-action pair.

This helped reduce the number of parameters compared to a fully tabular representation. It did worse than tabular SARSA(λ), but the error was certainly not horrible.



## Experiments/Results
I compared performance across different λ values using mean squared error against the Monte Carlo control estimate. I also plotted the optimal value function from Monte Carlo control in the same style as the blackjack exercise from Sutton and Barto. 
<img width="1276" height="958" alt="694F822F-47E3-4753-BD5E-267FF1B76E28" src="https://github.com/user-attachments/assets/3960f83b-31c5-4f6a-a4b2-a7dd27a6b549" />

<img width="1062" height="739" alt="0EF11E5B-4BED-4D76-85B3-8FCD880729AA_1_105_c" src="https://github.com/user-attachments/assets/a82327eb-6252-4590-90fe-60eb8765ea6c" />

<img width="1242" height="902" alt="AE3CE2A3-6F85-443B-BEF1-9DD9FC52C396" src="https://github.com/user-attachments/assets/15066bb9-4517-4b4a-86c5-b14e4a4debde" />





## How to run
Install dependencies:

```bash
pip install numpy matplotlib
```

Run the environment manually:

```bash
python3 environment.py
```

Run Monte Carlo control:

```bash
python3 montecarlo.py
```

Run tabular SARSA(lambda):

```bash
python3 sarsa.py
```

Run SARSA(lambda) with linear function approximation:

```bash
python3 approximation.py
```

Run tests:

```bash
python3 -m unittest test_easy21.py
```

Note: training scripts may take many minutes because they run many episodes.
