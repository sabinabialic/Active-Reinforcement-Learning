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
        
    
    def qLearning(self, discount_factor = 1.0, alpha = 0.6, epsilon = 0.1):
        for element in self.data:
            state = element[0]
            action = element[1]
                    
                # get probabilities of all actions from current state
                # action = self.policy(state)
                # take action and get reward, transit to next state
            value = self.qvalue(state, action)
            self.value[(state, action)] = value
            self.counter += 1


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

        # q = Q(state,action) + alpha(r(state) - Q(state, action) + gamma(numpy.max(Q(next state, next action) in range (possible action))))
        if (state, action) in self.value:
            q = self.value[(state, action)] + 0.1 * (r(state) - self.value[(state, action)] + 0.5 * max_value)
        else:
            q = 0.1 * (r(state) + 0.5 * max_value)
        print("Q-value: {} --- State: {} --- Action: {} --- Reward: {} \n".format(q, state, action, r(state)))

        return q


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

        print("Possible actions are: ", possibleActions)
        print("The maximum Q value is: ", max(allQs))

        # Return the optimal action under the learned policy
        return possibleActions[optimalMove]


t = td_qlearning('trajectory.csv')
print(t.value[("300000", "C")])
print(t.value[("300000", "R")])
print(t.value[("510100", "U")])
print(t.policy("201011")) 
print(t.policy("410000"))
print(t.policy("510100"))