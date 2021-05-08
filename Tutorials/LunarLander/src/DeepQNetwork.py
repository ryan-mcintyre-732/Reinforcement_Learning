import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class ReplayBuffer:

    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_memory = np.zeros(
            (self.mem_size, *input_dims),
            dtype=np.float32
        )

        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims),
            dtype=np.float32
        )

        self.action_memory = np.zeros(
            self.mem_size,
            dtype=np.int32
        )

        self.reward_memory = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

        self.terminal_memory = np.zeros(
            self.mem_size,
            dtype=np.int32
        )

    def store_transition(self, old_state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = old_state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        old_states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return old_states, actions, rewards, new_states, terminal


def build_dqn(learning_rate, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model


class Agent:
    def __init__(
            self, learning_rate, gamma, n_actions, epsilon_initial,
            batch_size, input_dims, epsilon_step=1e-4,
            epsilon_min=0.01, mem_size=1000000, filename='model.h5'
    ):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_step = epsilon_step
        self.batch_size = batch_size
        self.model_file = filename
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(learning_rate, n_actions, input_dims, 256, 256)

    def store_transition(self, old_state, action, reward, new_state, done):
        self.memory.store_transition(old_state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        old_states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(old_states)
        q_next = self.q_eval.predict(new_states)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

        self.q_eval.train_on_batch(old_states, q_target)

        self.epsilon = self.epsilon - self.epsilon_step \
            if self.epsilon > self.epsilon_min \
            else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
