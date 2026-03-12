import nn
import model
from qlearningAgents import PacmanQAgent
from backend import ReplayMemory
import layout
import copy

import numpy as np

class PacmanDeepQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 50000
        self.epsilon_explore = 1.0
        self.epsilon0 = 0.05
        self.epsilon = self.epsilon0
        self.discount = 0.9
        self.update_frequency = 3
        self.counts = None
        self.replay_memory = ReplayMemory(50000)
        self.min_transitions_before_training = 10000
        self.td_error_clipping = 1

        # Keep a fixed action-to-index mapping so each output neuron has a
        # consistent semantic meaning across all states.
        self.actions = ["North", "South", "East", "West", "Stop"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        # Initialize Q networks:
        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input
        self.state_dim = self.get_state_dim(layout_instantiated)
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ

    def _action_index(self, action):
        return self.action_to_idx[action]

    def get_state_dim(self, layout):
        pac_ft_size = 2
        ghost_ft_size = 2 * layout.getNumGhosts()
        food_capsule_ft_size = layout.width * layout.height
        return pac_ft_size + ghost_ft_size + food_capsule_ft_size

    def get_features(self, state):
        pacman_state = np.array(state.getPacmanPosition())
        ghost_state = np.array(state.getGhostPositions())
        capsules = state.getCapsules()
        food_locations = np.array(state.getFood().data).astype(np.float32)
        for x, y in capsules:
            food_locations[x][y] = 2
        return np.concatenate((pacman_state, ghost_state.flatten(), food_locations.flatten()))

    def initialize_q_networks(self, state_dim, action_dim=5):
        import model
        self.model = model.DeepQNetwork(state_dim, action_dim)
        self.target_model = model.DeepQNetwork(state_dim, action_dim)

    def getQValue(self, state, action):
        feats = self.get_features(state)
        state_node = nn.Constant(np.array([feats]).astype("float64"))
        q_values = self.model.run(state_node).data[0]
        action_index = self._action_index(action)
        return q_values[action_index]

    def shape_reward(self, reward):
        if reward > 100:
            reward = 10
        elif reward > 0 and reward < 10:
            reward = 2
        elif reward == -1:
            reward = 0
        elif reward < -100:
            reward = -10
        return reward

    def compute_q_targets(self, minibatch, network=None, target_network=None, doubleQ=False):
        """Prepare minibatches."""
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model

        states = np.vstack([x.state for x in minibatch])
        states = nn.Constant(states)
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        next_states = nn.Constant(next_states)
        done = np.array([x.done for x in minibatch])

        Q_predict = network.run(states).data
        Q_target = np.copy(Q_predict)

        state_indices = states.data.astype(int)
        state_indices = (state_indices[:, 0], state_indices[:, 1])
        exploration_bonus = 1 / (2 * np.sqrt((self.counts[state_indices] / 100)))

        replace_indices = np.arange(actions.shape[0])

        if doubleQ:
            # Double DQN: online network selects the action, target network evaluates it.
            next_action_indices = np.argmax(network.run(next_states).data, axis=1)
            next_q = target_network.run(next_states).data[replace_indices, next_action_indices]
        else:
            # Vanilla DQN: directly take max action value from target network.
            next_q = np.max(target_network.run(next_states).data, axis=1)

        target = rewards + exploration_bonus + (1 - done) * self.discount * next_q
        Q_target[replace_indices, actions] = target

        if self.td_error_clipping is not None:
            Q_target = Q_predict + np.clip(
                Q_target - Q_predict, -self.td_error_clipping, self.td_error_clipping
            )

        return Q_target

    def update(self, state, action, nextState, reward):
        action_index = self._action_index(action)
        done = nextState.isLose() or nextState.isWin()
        reward = self.shape_reward(reward)

        if self.counts is None:
            x, y = np.array(state.getFood().data).shape
            self.counts = np.ones((x, y))

        state = self.get_features(state)
        nextState = self.get_features(nextState)
        self.counts[int(state[0])][int(state[1])] += 1

        transition = (state, action_index, reward, nextState, done)
        self.replay_memory.push(*transition)

        if len(self.replay_memory) < self.min_transitions_before_training:
            self.epsilon = self.epsilon_start
        else:
            steps = self.update_amount - self.min_transitions_before_training
            frac = min(max(steps / float(self.epsilon_decay_steps), 0.0), 1.0)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
            minibatch = self.replay_memory.pop(self.model.batch_size)
            states = np.vstack([x.state for x in minibatch])
            states = nn.Constant(states.astype("float64"))
            Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)
            Q_target1 = nn.Constant(Q_target1.astype("float64"))
            self.model.gradient_update(states, Q_target1)

        if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
            self.target_model.set_weights(copy.deepcopy(self.model.parameters))

        self.update_amount += 1

    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)