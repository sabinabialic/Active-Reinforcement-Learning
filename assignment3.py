import numpy as np
import csv


class td_qlearning:
    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing

        with open(trajectory_filepath, 'r') as file:
            reader = csv.reader(file)
            self.data = list(reader)
        self.value = {}
        self.counter = 0
        self.qLearning()
        
    
    def qLearning(self):
        for element in self.data:
            state = element[0]
            action = element[1]
                    
            value = self.determine_qvalue(state, action)
            self.value[(state, action)] = value
            self.counter += 1


    def determine_qvalue(self, state, action):
        # returns the reward associatied with a state
        def r(state): return (-1 * state.count('1', 1, 6))

        options = []
        if self.counter + 1 < len(self.data)-1:
            next_state = self.data[self.counter+1][0]
            currentSquare = int(next_state[0])
            if currentSquare == 1: options = ['C', 'D']
            elif currentSquare == 2: options = ['C', 'R']
            elif currentSquare == 3: options = ['C', 'L', 'R', 'U', 'D']
            elif currentSquare == 4: options = ['C', 'L']
            elif currentSquare == 5: options = ['C', 'U']

            expected_values = []
            for option in options:
                if (next_state, option) in self.value:
                    expected_values.append(self.value[(next_state, option)])
                else:
                    expected_values.append(0)
            max_value = max(expected_values)
        else:
            max_value = 0.0

        if (state, action) in self.value:
            q = self.value[(state, action)] + 0.1 * (r(state) - self.value[(state, action)] + 0.5 * max_value)
        else:
            q = 0.1 * (r(state) + 0.5 * max_value)
        # print("Q-value: {} --- State: {} --- Action: {} --- Reward: {} \n".format(q, state, action, r(state)))

        return q

    
    def qvalue(self, state, action):
        return self.value[(state, action)]


    def policy(self, state):
        def currentPosition(state): return int(state[0])

        def moves(state):
            currentSquare = currentPosition(state)
            if currentSquare == 1: return ['C', 'D']
            elif currentSquare == 2: return ['C', 'R']
            elif currentSquare == 3: return ['C', 'L', 'R', 'U', 'D']
            elif currentSquare == 4: return ['C', 'L']
            elif currentSquare == 5: return ['C', 'U']

        # state is a string representation of a state
        possibleActions = moves(state)

        allQs = []
        for action in possibleActions:
            if (state, action) in self.value:
                allQs.append(self.value[(state, action)])
            else:
                allQs.append(0)

        optimalMove = allQs.index(max(allQs))

        return possibleActions[optimalMove]


t = td_qlearning('trajectory.csv')
