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

    def empty_memory(self):
        self.memory = []
        self.memory_index = 0

    def train(self):

    def clone(self):

