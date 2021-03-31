import numpy
import sys
import csv

class td_qlearning:
    alpha = 0.1
    gamma = 0.5

    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing
        with open(trajectory_filepath, 'r') as file:
            reader = csv.reader(file)
            self.data = list(reader)

        self.state = 'something'
        self.action = 'something'
        self.value = 'something'

    def qvalue(self, state, action):
        # returns the reward associatied with a state
        def r(state): return (-1 * state.count('1', 1, 6))
        # state is a string representation of a state
        # action is a string representation of an action
        print("This is the state: ", state)
        print("This is the action: ", action)
        print("The number of dirty squares is: ", state.count('1', 1, 6))

        q = 0 + 0.1*(r(state))
        print("The reward is: ", r(state))

        # q = Q(state,action) + alpha(r(state) - Q(state, action) + gamma(numpy.max(Q(next state, next action) in range (possible action))))

        # update the q value for the current state
        self.value = q

        # Return the q-value for the state-action pair
        return q

    def policy(self, state):
        # state is a string representation of a state

        # Return the optimal action under the learned policy
        return a


t = td_qlearning('trajectory.csv')
print("The q value is: ", t.qvalue(t.data[3][0], t.data[3][1]))
