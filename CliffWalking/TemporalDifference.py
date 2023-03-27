
import sys
import gymnasium as gym
import numpy as np
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

def update_Q_sarsa(alpha, gamma, Q, epsilon, state, action, reward, next_state=None, next_action=None):
    Q_next = Q[next_state][next_action] if next_state is not None else 0
    Q_curr = Q[state][action]
    return Q_curr + alpha * (reward + gamma * Q_next - Q_curr)

def update_Q_sarsamax(alpha, gamma, epsilon, Q, state, action, reward, next_state=None, next_action=None):
    greedy_action = np.argmax(Q[next_state])
    Q_curr = Q[state][action]
    return Q_curr + alpha * (reward + gamma * Q[next_state][greedy_action] - Q_curr)

def update_Q_expected_sarsa(alpha, gamma, epsilon, Q, state, action, reward, next_state=None, next_action=None):
    # random action by default (epsilon)
    policy_s = np.ones(env.nA) * epsilon / env.nA
    # greedy action (1 - epsilon)
    policy_s[np.argmax(Q[next_state])] = 1 - epsilon + (epsilon / env.nA)
    Q_next = np.dot(Q[next_state], policy_s)
    Q_curr = Q[state][action]
    return Q_curr + alpha * (reward + gamma * Q_next - Q_curr)

def epsilon_greedy(Q, state, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.choice(np.arange(env.nA))

plot_every = 100

def TD_method(update_method, env, num_episode, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.nA))
    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episode)   # average scores over every plot_every episodes

    for i in range(1, num_episode+1):
        if i % 100 == 0:
            print("\rEpisode {}/{}".format(i, num_episode), end="")
            sys.stdout.flush()   

        score = 0                                             # initialize score
        state, _ = env.reset()                                   # start episode
        epsilon = 1.0 / i                                     # set value of epsilon
        conti = True
        action = epsilon_greedy(Q, state, epsilon)
        while conti:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            if not done:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                Q[state][action] = update_method(alpha, gamma, epsilon, Q, state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            else:
                Q[state][action] = update_method(alpha, gamma, epsilon, Q, state, action, reward)
                tmp_scores.append(score)
                conti = False
        
        if (i % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))
    
    # plot performance
    plt.plot(np.linspace(0,num_episode,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores)) 
    return Q

def sarsa(env, num_episode, alpha, gamma=1.0):
    return TD_method(update_Q_sarsa, env, num_episode, alpha, gamma)

def Q_learning(env, num_episode, alpha, gamma=1.0):
    return TD_method(update_Q_sarsamax, env, num_episode, alpha, gamma)

def expected_sarsa(env, num_episode, alpha, gamma=1.0):
    return TD_method(update_Q_expected_sarsa, env, num_episode, alpha, gamma)

"""
Q_sarsa = sarsa(env, 5000, .01)
Q_sarsa = Q_learning(env, 5000, .01)
"""
Q_sarsa = expected_sarsa(env, 5000, .01)
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)

check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)


