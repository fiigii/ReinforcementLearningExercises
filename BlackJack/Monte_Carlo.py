import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

def generate_episode_from_limit_stochastic(blackjac_env):
    episode = []
    state, _ = blackjac_env.reset()
    conti = True
    while conti:
        # Action: Stick 0 or Hit 1
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, _, _ = blackjac_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        conti = not done
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    expected_return = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q= defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # every-visit avarage
        for i, state in enumerate(states):
            expected_return[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = expected_return[state][actions[i]] / N[state][actions[i]]
    return Q

env = gym.make('Blackjack-v1')


"""
# obtain the action-value function
#Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)
"""

def epsilon_greedy_probs(action_values, epsilon, sizeA):
    """ returns possiblities of [stick, hit] """
    # randomly select (exploration) by default
    policy_s = np.ones(sizeA) * epsilon / sizeA
    best_a = np.argmax(action_values) # the index of the best action in action_values
    # select the best action (exploitation)
    policy_s[best_a] = 1 - epsilon + (epsilon / sizeA)
    return policy_s
    

def generate_episode_from_Q(env, Q, epsilon, sizeA):
    episode = []
    state, _ = env.reset()
    conti = True
    get_action = lambda state: (
        np.random.choice(np.arange(sizeA), p=epsilon_greedy_probs(Q[state], epsilon, sizeA)) \
            if state in Q else env.action_space.sample()
    )
    while conti:
        action = get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        conti = not done
    return episode

def update_Q(env, episode, Q, alpha, gamma=1.0):
    states, actions, rewards = zip(*episode)
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_q = Q[state][actions[i]]
        expected_return = (sum(rewards[i:]*discounts[:-(1+i)]))
        Q[state][actions[i]] = old_q * (1-alpha) + alpha * expected_return
    return Q


def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    sizeA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(sizeA))
    epsilon = eps_start
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = max(epsilon*eps_decay, eps_min)
        episode = generate_episode_from_Q(env, Q, epsilon, sizeA)
        Q = update_Q(env, episode, Q, alpha, gamma)
    # compute the optimal policy from the optimial action-value function
    # state -> list(action,value) => state -> action (with the best value)
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q

policy, Q = mc_control(env, 500000, 0.02)

V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

plot_policy(policy)