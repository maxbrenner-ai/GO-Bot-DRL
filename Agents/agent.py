class Agent:
    def __init__(self):
        pass

    def output_action(self, user_frame):
        database_return = self.state_tracker.query_database(user_frame)
        self.state_tracker.update(user_frame, database_return)
        current_state = self.state_tracker.return_state()

        return self.policy_learner.return_action(current_state)

    def get_action(self):
