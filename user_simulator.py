class UserSimulator:
    def __init__(self):
        self.agenda = UserAgenda()
        self.internal_state = ...

    def reset(self, goal_list):
        self._pick_goal(goal_list)
        self._refresh_agenda()

    # Randomly choose a goal from the list of available user goals
    def _pick_goal(self, goal_list):
        goal_index = ...
        self.goal = UserGoal()

    def step(self, agent_frame):
        return next_user_frame, reward, done, succ

    def _refresh_agenda(self):
        self.agenda.refresh()

    def _reward_function(self, num):
        return reward


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

