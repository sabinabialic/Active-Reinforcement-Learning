import numpy
import sys
import csv

class td_qlearning:
    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing
        with open(trajectory_filepath, 'r') as file:
            reader = csv.reader(file)
            self.data = list(reader)

        self.value  = 0

    def qvalue(self, state, action):
        # returns the reward associatied with a state
        def r(state): return (-1 * state.count('1', 1, 6))

        # returns the next state
        def next(state, action):
            pos = int(state[0])
            # case where the current square is dirty and the action is clean
            if ("C" in action): return state[:pos] + "0" + state[pos+1:]
            # case where the action is up
            elif ("U" in action):
                if (pos == 3): return "1" + state[1:6]
                if (pos == 5): return "3" + state[1:6]
            # case where the action is down
            elif ("D" in action):
                if (pos == 1): return "3" + state[1:6]
                if (pos == 3): return "5" + state[1:6]
            # case where the action is right
            elif ("R" in action):
                if (pos == 2): return "3" + state[1:6]
                if (pos == 3): return "4" + state[1:6]
            # case where the action is left
            elif ("L" in action):
                if (pos == 4): return "3" + state[1:6]
                if (pos == 3): return "1" + state[1:6]

        # state is a string representation of a state
        # action is a string representation of an action
        print("This is the state: ", state)
        print("This is the action: ", action)
        print("The number of dirty squares is: ", state.count('1', 1, 6))
        print("The next state is: ", next(state, action))

        # q = Q(state,action) + alpha(r(state) - Q(state, action) + gamma(numpy.max(Q(next state, next action) in range (possible action))))
        q = self.value + 0.1*(r(state) - self.value) #+ 0.5(numpy.max(self.qvalue(state, allActions[1])))

        print("The reward is: ", r(state))

        # update the q value for the current state
        self.value = q

        # Return the q-value for the state-action pair
        return q

    def policy(self, state):
        def currentPosition(state): return int(state[0])

        def moves(state):
            currentSquare = currentPosition(state)
            if currentSquare == 1: return ['C', 'D']
            elif currentSquare == 2: return ['C', 'R']
            elif currentSquare == 3: return ['C', 'L', 'R', 'U', 'D']
            elif currentSquare == 4: return ['C', 'L']
            elif currentSquare == 5: ['C', 'U']

        # state is a string representation of a state
        possibleActions = moves(state)

        allQs = []
        for i in range(len(possibleActions)):
            allQs.append(t.qvalue(state, possibleActions[i]))

        optimalMove = allQs.index(max(allQs))

        print("Possible actions are: ", possibleActions)
        print("The maximum Q value is: ", max(allQs))

        # Return the optimal action under the learned policy
        return possibleActions[optimalMove]


t = td_qlearning('trajectory.csv')
print("The q value is: ", t.qvalue(t.data[3][0], t.data[3][1]))
print("The policy is: ", t.policy(t.data[3][0]))
