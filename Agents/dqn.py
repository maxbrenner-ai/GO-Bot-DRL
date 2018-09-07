import constants as C


class DQNAgent:
    def __init__(self):
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = C['max_mem_size']

        self._make_graph()

    def _make_graph(self):

    def get_action(self, state):

    def add_experience(self, experience):
        if self.memory_index == self.max_memory_size:
            self.memory_index = 0
        if len(self.memory) != self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = experience
        self.memory_index += 1

    def empty_memory(self):
        self.memory = []
        self.memory_index = 0

    def train(self):

    def clone(self):

