import pandas as pd
import numpy as np
NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1


class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): # You can add more arguments if you want
        self.name = "agent1"
        self.actions = list(range(num_actions))
        self.states = num_states
        self.discount = discount
        self.lr = learning_rate
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64) # create the q tab

    def process_experience(self, state, action, next_state, reward, done): # You can add more arguments if you want
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        
        if next_state != 'terminal':
            q_target = reward + self.discount * self.q_table.loc[next_state, :].max()
        else :
            q_target = reward
            
        self.q_table.loc[state, action] = (1 - self.lr) * self.q_table.loc[state, action] + self.lr * (q_target - q_predict)

        pass

    def select_action(self, state): # You can add more arguments if you want
        """
        Returns an action, selected based on the current state
        """
        self.check_state_exist(state)
        
        if np.random.uniform() < EPSILON:
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else :
            action = np.random.choice(self.actions)
        
        return action
        pass

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")
