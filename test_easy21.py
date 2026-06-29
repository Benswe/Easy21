import os
import unittest
from collections import defaultdict
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/easy21_matplotlib")

from environment import easy21Env
from montecarlo import epsilon_greedy_mc, trajectory, monte_carlo
from sarsa import MSE, sarsa_lambda


# Tiny deterministic environment for fast algorithm smoke tests.
class OneStepEnv:
    def __init__(self, reward=1):
        self.actions = ["hit", "stick"]
        self.reward = reward
        self.state = (1, 10)

    def reset(self):
        return self.state

    def step(self, action):
        return self.state, self.reward, True


# Fixed policy used when we do not want randomness in a test.
def always_stick(env, Q, state, N_s):
    return "stick"


class TestEasy21Environment(unittest.TestCase):
    # These tests pin down the Easy21 game rules.
    def test_reset_deals_black_cards(self):
        env = easy21Env()

        with patch("environment.random.randint", side_effect=[3, 7]):
            state = env.reset()

        self.assertEqual(state, (3, 7))
        self.assertFalse(env.done)

    def test_draw_card_black_two_thirds_branch(self):
        env = easy21Env()

        with patch("environment.random.random", return_value=0.9):
            with patch.object(env, "draw_black_card", return_value=5):
                self.assertEqual(env.draw_card(), 5)

    def test_draw_card_red_one_third_branch(self):
        env = easy21Env()

        with patch("environment.random.random", return_value=0.1):
            with patch.object(env, "draw_black_card", return_value=5):
                self.assertEqual(env.draw_card(), -5)

    def test_hit_continues_when_player_is_in_bounds(self):
        env = easy21Env()
        env.dealer_card = 2
        env.player_sum = 10

        with patch.object(env, "draw_card", return_value=5):
            next_state, reward, done = env.step("hit")

        self.assertEqual(next_state, (2, 15))
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_hit_loses_when_player_busts(self):
        env = easy21Env()
        env.dealer_card = 2
        env.player_sum = 10

        with patch.object(env, "draw_card", return_value=12):
            next_state, reward, done = env.step("hit")

        self.assertEqual(next_state, (2, 22))
        self.assertEqual(reward, -1)
        self.assertTrue(done)

    def test_stick_wins_when_dealer_busts(self):
        env = easy21Env()
        env.dealer_card = 16
        env.player_sum = 10

        with patch.object(env, "draw_card", return_value=6):
            next_state, reward, done = env.step("stick")

        self.assertEqual(next_state, (22, 10))
        self.assertEqual(reward, 1)
        self.assertTrue(done)

    def test_stick_compares_final_totals(self):
        env = easy21Env()
        env.dealer_card = 17
        env.player_sum = 18
        self.assertEqual(env.step("stick"), ((17, 18), 1, True))

        env = easy21Env()
        env.dealer_card = 20
        env.player_sum = 18
        self.assertEqual(env.step("stick"), ((20, 18), -1, True))

        env = easy21Env()
        env.dealer_card = 18
        env.player_sum = 18
        self.assertEqual(env.step("stick"), ((18, 18), 0, True))

    def test_invalid_action_raises(self):
        env = easy21Env()

        with self.assertRaises(ValueError):
            env.step("dance")


class TestPoliciesAndAlgorithms(unittest.TestCase):
    # These tests keep the learning helpers deterministic and quick.
    def test_epsilon_greedy_exploits_best_action_when_not_exploring(self):
        env = OneStepEnv()
        Q = defaultdict(float)
        state = env.state
        Q[(state, "hit")] = -1
        Q[(state, "stick")] = 1
        N_s = defaultdict(int)
        N_s[state] = 10_000

        with patch("montecarlo.random.random", return_value=0.99):
            action = epsilon_greedy_mc(env, Q, state, N_s)

        self.assertEqual(action, "stick")

    def test_trajectory_collects_state_action_reward_until_done(self):
        env = OneStepEnv(reward=1)
        Q = defaultdict(float)
        N_s = defaultdict(int)

        result = trajectory(Q, env, always_stick, N_s)

        self.assertEqual(result, [((1, 10), "stick", 1)])

    def test_monte_carlo_updates_one_step_return(self):
        env = OneStepEnv(reward=1)

        Q = monte_carlo(env, always_stick, num_episodes=3)

        self.assertEqual(Q[((1, 10), "stick")], 1)

    def test_sarsa_lambda_updates_one_step_return(self):
        env = OneStepEnv(reward=1)

        Q = sarsa_lambda(env, always_stick, llambda=0.5, num_episodes=3)

        self.assertEqual(Q[((1, 10), "stick")], 1)

    def test_mse_can_return_sum_or_average(self):
        env = easy21Env()
        Q = defaultdict(float)
        Q_star = defaultdict(lambda: 1.0)

        self.assertEqual(MSE(env, Q, Q_star, average=False), 420)
        self.assertEqual(MSE(env, Q, Q_star, average=True), 1)


if __name__ == "__main__":
    unittest.main()
