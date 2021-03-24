import numpy
import sys

class td_qlearning:

  alpha = 0.1
  gamma = 0.5

  def __init__(self, trajectory_filepath):
    # trajectory_filepath is the path to a file containing a trajectory through state space
    # Return nothing
    self.nextState = "something"
    self.nextAction = "something"

  def qvalue(self, state, action):
    def r(state): return (-1 * s.count('1', 1, 5))
    # state is a string representation of a state
    # action is a string representation of an action

    # q = Q(state,action) + alpha(r(state) + gamma(Q(next state, next action) - Q(state, action)))

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state

    # Return the optimal action under the learned policy
    return a
