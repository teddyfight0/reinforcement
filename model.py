import nn
import numpy as np

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.00025
        self.numTrainingGames = 10000
        self.batch_size = 32
        self.hidden_size = 256
        self.W1 = nn.Parameter(state_dim, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        
        self.W2 = nn.Parameter(self.hidden_size, action_dim)
        self.b2 = nn.Parameter(1, action_dim)
        
        self.parameters = [self.W1, self.b1, self.W2, self.b2]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        # 运行网络得到Q值预测
        Q_predicted = self.run(states)
        
        # 计算平方损失
        loss = nn.SquareLoss(Q_predicted, Q_target)
        
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        # 前向传播：两层神经网络
        # 第一层：线性变换 + bias + ReLU激活
        z1 = nn.AddBias(nn.Linear(states, self.W1), self.b1)
        a1 = nn.ReLU(z1)
        
        # 第二层（输出层）：线性变换 + bias（不需要激活函数）
        z2 = nn.AddBias(nn.Linear(a1, self.W2), self.b2)
        
        return z2

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        # 计算损失
        loss = self.get_loss(states, Q_target)
        
        # 计算梯度
        gradients = nn.gradients(loss, self.parameters)
        
        # 更新参数
        for param, grad in zip(self.parameters, gradients):
            param.update(grad, -self.learning_rate)
