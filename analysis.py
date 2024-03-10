# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # Need to lure agent to cross the risky bridge for a higher reward
    # We can either increase the discount to make the agent value future rewards
    # Or we can decrease the noise to increase predictability (I did this)
    answerDiscount = 0.9
    answerNoise = 0.002
    return answerDiscount, answerNoise


def question3a():
    # Low, low, super low
    answerDiscount = 0.2  # Low discount to prefer immediate gratification
    # Uncertainty of actions
    answerNoise = 0.05  # Low to decrease weariness
    # Reward for living
    answerLivingReward = -5.0  # Pacman doesn't want to live :(
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    # Low noise, discount, and LOW living reward
    answerDiscount = 0.2  # Low to focus on nearest exit
    answerNoise = 0.2  # Low to avoid fear of cliff
    answerLivingReward = -1.5  # Pacman still would rather not be around
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3c():
    # High, high, low
    answerDiscount = 0.9  # High to value distant exits (explore)
    answerNoise = 0.2  # Low to avoid fear of cliff
    answerLivingReward = -1.5  # :((
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    # High, low, low
    answerDiscount = 1.0  # High for distant exits (explore)
    answerNoise = 0.1  # Low to avoid cliff
    answerLivingReward = -0.7  # Gettin' betta
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    # Low, low, high
    answerDiscount = 0.1  # Low to devalue any exits
    answerNoise = 0.0  # Doesn't matter, living is the only relevant factor
    answerLivingReward = 15.0  # YAY PACMAN
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question8():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
