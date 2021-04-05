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
        # state is a string representation of a state
        # action is a string representation of an action
        print("This is the state: ", state)
        print("This is the action: ", action)
        print("The number of dirty squares is: ", state.count('1', 1, 6))

        q = self.value + 0.1*(r(state) - self.value) #+ 0.5(numpy.max(self.qvalue(state, allActions[1])))

        print("The reward is: ", r(state))

        # q = Q(state,action) + alpha(r(state) - Q(state, action) + gamma(numpy.max(Q(next state, next action) in range (possible action))))

        # update the q value for the current state
        self.value = q

        # Return the q-value for the state-action pair
        return q

    def policy(self, state):
        def currentPosition(state): return state[0]

        def moves(state):
            currentSquare = currentPosition(state)
            if currentSquare == 1: return ['C', 'D']
            elif currentSquare == 2: return ['C', 'R']
            elif currentSquare == 3: return ['C', 'L', 'R', 'U', 'D']
            elif currentSquare == 4: return ['C', 'L']
            return ['C', 'U']

        # state is a string representation of a state
        possibleActions = moves(state)

        for i in possibleActions:
            # Does not work
            allQs.append(qvalue(state, possibleActions[i]))

        optimalMove = allQs.index(max(allQs))

        print("Possible actions are: ", possibleActions)
        print("The maximum Q value is: ", max(allQs))

        # Return the optimal action under the learned policy
        return possibleActions[optimalMove]


t = td_qlearning('trajectory.csv')
print("The q value is: ", t.qvalue(t.data[3][0], t.data[3][1]))
print("The policy is: ", t.policy(t.data[3][0]))
