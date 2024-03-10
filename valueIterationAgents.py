# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Iterate for specified number
        for _ in range(self.iterations):
            # Temp copy of values to store updates per iteration
            tempValues = self.values.copy()

            # Loop to iterate through all states in MDP
            for state in self.mdp.getStates():
                # Skip any terminals, their value = 0
                if not self.mdp.isTerminal(state):
                    # Update state value to max Q
                    tempValues[state] = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))

            # Update values for this iteration
            self.values = tempValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Init qValue to 0 for every new state-action
        qValue = 0

        # Loop of all possible states given the state-action
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Calc reward for transition
            reward = self.mdp.getReward(state, action, nextState)
            # Update qValue with expected utility of the transition
            qValue += prob * (reward + self.discount * self.values[nextState])

        # Return updated qValue for each state-action
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Unless curr state is terminal, we return the action with the highest qValue from whatever state we are in
        if self.mdp.isTerminal(state):
            return None
        # Max returns the action ([1]) from the (qValue, action) tuple
        return max((self.computeQValueFromValues(state, action), action) for action in self.mdp.getPossibleActions(state))[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Update value of one state per iteration in a fixed order
        # If terminal, we skip still
        states = self.mdp.getStates()  # Get every state
        numStates = len(states)  # How many states?

        # Loop for iterations
        for i in range(self.iterations):
            # What is the state we are updating?
            updateState = states[i % numStates]
            # If the update is terminal, skip it
            if not self.mdp.isTerminal(updateState):
                # New value, same calc as Q1
                bestVal = max(self.computeQValueFromValues(updateState, action) for action in self.mdp.getPossibleActions(updateState))

                self.values[updateState] = bestVal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Need a pQueue to store states. This will store how much we expect values to change.
        pQueue = util.PriorityQueue()

        # Let's find all predecessors
        preds = {}
        for state in self.mdp.getStates():
            # filter out terminals
            if not self.mdp.isTerminal(state):
                # get actions
                for action in self.mdp.getPossibleActions(state):
                    # iterate
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        # if we haven't recorded preds, do it!
                        if nextState not in preds:
                            # add to set
                            preds[nextState] = set()
                        preds[nextState].add(state)

        # Record priorities
        for state in self.mdp.getStates():
            # filter out terminals
            if not self.mdp.isTerminal(state):
                # same equation as Q1, Q4
                bestVal = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                difference = abs(self.values[state] - bestVal)  # Record diff in values
                pQueue.update(state, -difference)  # pQueue is min heap ;)

        # Update pQueue
        for _ in range(self.iterations):
            # check if pQueue is empty
            if pQueue.isEmpty():
                break

            # pop off state we're on
            state = pQueue.pop()

            # filter out terminals
            if not self.mdp.isTerminal(state):
                # Update value (same as above)
                self.values[state] = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))

            # update priorities for all preds
            for pred in preds.get(state, []):  # [] is priority part of tuple
                # filter out terminals
                if not self.mdp.isTerminal(pred):
                    # Same, but a lil different (use pred)
                    bestVal = max(self.computeQValueFromValues(pred, action) for action in self.mdp.getPossibleActions(pred))
                    difference = abs(self.values[pred] - bestVal)
                    if difference > self.theta:
                        pQueue.update(pred, -difference)  # MIN HEAP