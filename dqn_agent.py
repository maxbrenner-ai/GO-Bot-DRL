import constants as C
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
from dialogue_config import rule_requests, agent_actions


# Some of code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

# Note: They do not anneal epsilon
class DQNAgent:
    def __init__(self, state_size):
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = C['max_mem_size']
        self.eps = C['epsilon_init']  # Note they do not anneal eps, and default is 0, so i should test this out
        self.num_actions = C['num_actions']
        self.lr = C['learning_rate']
        self.gamma = C['gamma']
        self.num_batches = C['num_batches']
        self.batch_size = C['batch_size']

        self.state_size = state_size

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()
        self.none_state = np.zeros(self.state_size)

        self.possible_actions = agent_actions

        self.reset()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def reset(self):
        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'
        self.rule_request_set = rule_requests

    def get_action(self, state, use_rule=False):
        if self.eps > random.random():
            return random.randint(0, self.num_actions - 1)
        else:
            if use_rule:
                return self._rule_predict()
            else:
                return np.argmax(self._dqn_predict_one(state))

    def _rule_predict(self):
        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': "inform", 'inform_slots': {'match_found': 'PLACEHOLDER'}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            assert True is False

        return self._action_index(rule_response)

    # For the rule policy to index the crafted response
    def _action_index(self, response):
        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i

    def _dqn_predict(self, states, target=False):
        if target:
            return self.tar_model.predict(states)
        else:
            return self.beh_model.predict(states)

    def _dqn_predict_one(self, state, target=False):
        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def add_experience(self, state, action, reward, next_state, done):

        assert isinstance(action, int)

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        self.memory = []
        self.memory_index = 0

    # States, actions, rewards, next_states, done
    def train(self):
        for _ in range(self.num_batches):
            batch = self._sample_memory(self.batch_size)
            batch_size = len(batch)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([self.none_state if sample[4] is True else sample[3] for sample in batch])

            beh_state_preds = self._dqn_predict(states)  # For leveling error
            beh_next_states_preds = self._dqn_predict(next_states)  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict(next_states, target=True)  # For target value for DQN

            inputs = np.zeros((batch_size, self.state_size))
            targets = np.zeros((batch_size, self.num_actions))

            # Todo: Test if this enum works
            for i, (s, a, r, s_, d) in enumerate(batch_size):
                t = beh_state_preds[i]
                t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.train_on_batch(inputs, targets)

    def _sample_memory(self, num):
        num = min(num, len(self.memory))
        return random.sample(self.memory, num)

    def copy(self):
        self.tar_model.set_weights(self.beh_model.get_weights())
