import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Step = 1
        self.alpha = 0.2
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
    def epsilon_greedy(self, state, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.Step += 1
        return self.epsilon_greedy(state, 1.0/self.Step)
    
    def update_Q_sarsamax(self, epsilon, state, action, reward, next_state):
        greedy_action = np.argmax(self.Q[next_state])
        Q_curr = self.Q[state][action]
        return Q_curr + self.alpha * (reward + self.Q[next_state][greedy_action] - Q_curr)
    
    def update_Q_expected_sarsa(self, epsilon, state, action, reward, next_state):
        # random action by default (epsilon)
        policy_s = np.ones(self.nA) * epsilon / self.nA
        # greedy action (1 - epsilon)
        policy_s[np.argmax(self.Q[next_state])] = 1 - epsilon + (epsilon / self.nA)
        Q_next = np.dot(self.Q[next_state], policy_s)
        Q_curr = self.Q[state][action]
        return Q_curr + self.alpha * (reward + Q_next - Q_curr)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_Q_expected_sarsa(1.0/self.Step, state, action, reward, next_state)