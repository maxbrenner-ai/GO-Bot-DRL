class Agent:
    def __init__(self):
        self.state_tracker = DialogueStateTracker()
        self.policy_learner = PolicyLearner()

    def output_action(self, user_frame):
        database_return = self.state_tracker.query_database(user_frame)
        self.state_tracker.update(user_frame, database_return)
        current_state = self.state_tracker.return_state()

        return self.policy_learner.return_action(current_state)

    def train(self):

class DialogueStateTracker:
    def __init__(self):
        internal_state = ...

    # Query the database given the user request
    def query_database(self, frame):

    # Update its internal state
    def update(self, frame, database_output):

    # create and return St for the policy learner
    def return_state(self):

class PolicyLearner:
    def __init__(self):
        policy = ...

    def return_action(self, state):