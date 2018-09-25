import random
from dialogue_config import usersim_default_slot


class UserSimulator:
    def __init__(self, goal_list):
        self.goal_list = goal_list

        # This is eqivalent to ticket in the moviedatabase, it MUST be in req of user sim goal
        self.default_slot = usersim_default_slot
        self.agenda = UserAgenda()
        self.internal_state = ...

    def reset(self):
        self.goal = random.choice(self.goal_list)

    # Randomly choose a goal from the list of available user goals
    def _pick_goal(self):
        goal_index = ...
        self.goal = UserGoal()

    def step(self, agent_frame, round_num):
        # First check round num
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

