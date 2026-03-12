import nn
import numpy as np

class DeepQNetwork():
    """
    Deep Q-value Network (DQN) with three hidden layers to approximate Q(s,a).
    """
    def __init__(self, state_dim, action_dim):
        self.learning_rate = 0.0001
        self.numTrainingGames = 8600
        self.batch_size = 128
        self.hidden_size1 = 512
        self.hidden_size2 = 256
        self.hidden_size3 = 128
        self.print_every = 2000
        self.update_steps = 0

        self.W1 = nn.Parameter(state_dim, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)

        self.W2 = nn.Parameter(self.hidden_size1, self.hidden_size2)
        self.b2 = nn.Parameter(1, self.hidden_size2)

        self.W3 = nn.Parameter(self.hidden_size2, self.hidden_size3)
        self.b3 = nn.Parameter(1, self.hidden_size3)

        self.W4 = nn.Parameter(self.hidden_size3, action_dim)
        self.b4 = nn.Parameter(1, action_dim)

        self.parameters = [self.W1, self.b1,
                           self.W2, self.b2,
                           self.W3, self.b3,
                           self.W4, self.b4]

    def run(self, states):
        """
        Forward pass for a batch of states.
        """
        z1 = nn.AddBias(nn.Linear(states, self.W1), self.b1)
        a1 = nn.ReLU(z1)

        z2 = nn.AddBias(nn.Linear(a1, self.W2), self.b2)
        a2 = nn.ReLU(z2)

        z3 = nn.AddBias(nn.Linear(a2, self.W3), self.b3)
        a3 = nn.ReLU(z3)

        z4 = nn.AddBias(nn.Linear(a3, self.W4), self.b4)
        return z4

    def set_weights(self, layers):
        self.parameters = []
        for layer in layers:
            self.parameters.append(layer)

    def get_loss(self, states, Q_target):
        """
        Squared loss between predicted Q-values and targets.
        """
        Q_predicted = self.run(states)
        return nn.SquareLoss(Q_predicted, Q_target)

    def gradient_update(self, states, Q_target):
        """
        One gradient descent step.
        """
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)
        for param, grad in zip(self.parameters, gradients):
            param.update(grad, -self.learning_rate)

        self.update_steps += 1
        if self.update_steps % self.print_every == 0:
            loss_scalar = nn.as_scalar(loss)
            grad_norms = [np.linalg.norm(g.data) for g in gradients]
            max_grad = max(grad_norms) if grad_norms else 0.0
            print(f"[DQN] step={self.update_steps:6d} "
                  f"loss={loss_scalar:8.4f} "
                  f"max_grad={max_grad:8.4f} "
                  f"lr={self.learning_rate}", flush=True)