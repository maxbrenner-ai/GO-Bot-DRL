class StateTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0

    def get_state(self):
        ...

    def update_state_agent(self, agent_action):
        self.turn += 1

    def update_state_user(self, user_action):
        self.turn += 1
