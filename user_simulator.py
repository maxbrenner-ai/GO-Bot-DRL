class UserSimulator:
    def __init__(self):
        self.agenda = UserAgenda()
        self.internal_state = ...

    def reset(self, goal_list):
        self.pick_goal(goal_list)
        self.refresh_agenda()

    # Randomly choose a goal from the list of available user goals
    def pick_goal(self, goal_list):
        goal_index = ...
        self.goal = UserGoal()

    def refresh_agenda(self):
        self.agenda.refresh()

    def output_action(self, agent_frame):
        # (if agent_frame none then start of conversation)
        # Make action based off of agenda and goal (given agent request and current state)
        action = ...
        # update internal state
        self.internal_state = ...
        # return action picked action
        return action

    def check_conv_success(self):
        if len(self.agenda.stack) != 0:
            return False
        # Now check constraints


class UserGoal:
    def __init__(self):
        inform_slots = []  # What info it already has
        request_slots = []  # What info it needs
        # Req and optional overlap with inform and request
        required_slots = []
        optional_slots = []


class UserAgenda:
    def __init__(self):
        self.stack = []

    def refresh(self):

