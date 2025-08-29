# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # 初始化Q值表，使用Counter来存储Q(state, action)值
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # 返回Q(state, action)的值，如果从未见过该状态-动作对，返回0.0
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # 获取该状态的所有合法动作
        legalActions = self.getLegalActions(state)
        
        # 如果没有合法动作（终止状态），返回0.0
        if not legalActions:
            return 0.0
        
        # 返回所有合法动作中Q值最大的那个
        maxQValue = float('-inf')
        for action in legalActions:
            qValue = self.getQValue(state, action)
            maxQValue = max(maxQValue, qValue)
        
        return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 获取该状态的所有合法动作
        legalActions = self.getLegalActions(state)
        
        # 如果没有合法动作（终止状态），返回None
        if not legalActions:
            return None
        
        # 找到Q值最大的动作
        bestAction = None
        bestQValue = float('-inf')
        
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action
        
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        # 如果没有合法动作，返回None
        if not legalActions:
            return None
        
        # epsilon-贪心策略：以epsilon概率随机选择动作，否则选择最优动作
        if util.flipCoin(self.epsilon):
            # 随机选择动作
            action = random.choice(legalActions)
        else:
            # 选择最优动作
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q-Learning更新公式：
        # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        # 获取当前Q值
        currentQ = self.getQValue(state, action)
        
        # 获取下一状态的最大Q值
        nextStateValue = self.computeValueFromQValues(nextState)
        
        # 计算新的Q值
        newQ = currentQ + self.alpha * (reward + self.discount * nextStateValue - currentQ)
        
        # 更新Q值表
        self.qValues[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # 获取特征向量
        features = self.featExtractor.getFeatures(state, action)
        
        # 计算Q值：Q(s,a) = w · f(s,a) = Σ w_i * f_i(s,a)
        qValue = 0.0
        for feature, value in features.items():
            qValue += self.weights[feature] * value
        
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # 近似Q-Learning权重更新公式：
        # 对于每个特征 i: w_i ← w_i + α * difference * f_i(s,a)
        # 其中 difference = [r + γ max Q(s',a') - Q(s,a)]
        
        # 获取特征向量
        features = self.featExtractor.getFeatures(state, action)
        
        # 计算当前Q值
        currentQ = self.getQValue(state, action)
        
        # 计算下一状态的最大Q值
        nextStateValue = self.computeValueFromQValues(nextState)
        
        # 计算差值（TD error）
        difference = reward + self.discount * nextStateValue - currentQ
        
        # 更新每个特征的权重
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
