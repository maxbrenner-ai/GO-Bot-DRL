import random
from dialogue_config import usersim_default_slot
import constants as C
import random


class UserSimulator:
    def __init__(self, goal_list):
        self.goal_list = goal_list

        self.max_round = C['max_round_num']
        # This is eqivalent to ticket in the moviedatabase, it MUST be in req of user sim goal
        self.default_slot = usersim_default_slot

    def reset(self):
        self.state = {}
        # Add all inform slots informed by agent or usersim to this dict
        self.state['history_slots'] = {}
        # Any inform slots for the current usersim action, empty at start of turn
        self.state['inform_slots'] = {}
        # Current request slots the usersim wants to request
        self.state['request_slots'] = {}
        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state['rest_slots'] = {}
        self.state['intent'] = ''
        self.goal = random.choice(self.goal_list)
        # Add default slot to requests of goal
        self.goal['request_slots'][self.default_slot] = 'UNK'
        # False for failure, true for success, auto init to failure
        self.constraint_check = False

        # Init rest slots: Format {'slot name': 'request' or 'inform', ...}
        for req_slot in self.goal['request_slots']:
            self.state['rest_slots'][req_slot] = 'request'
        for inf_slot in self.goal['inform_slots']:
            self.state['rest_slots'][inf_slot] = 'inform'

    def _return_init_action(self):
        # Always request
        self.state['intent'] = 'request'

        ...

    def step(self, agent_action, round_num):
        # Add all inform slots just informed to history slots and Clear all state inform slots (just sent out)
        self.state['history_slots'].update(self.state['inform_slots'])
        self.state['inform_slots'].clear()

        done = False
        succ = None  # False means loss, true means win, none means not done
        # First check round num, if past max then fail
        if round_num > self.max_round:
            done = True
            succ = False
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            agent_intent = agent_action['intent']
            if agent_intent == 'request':
                self.response_to_request(agent_action)
            if agent_intent == 'inform':
                self.response_to_inform(agent_action)
            if agent_intent == 'match_found':
                self.response_to_match_found(agent_action)
            if agent_intent == 'done':
                self.response_to_done(agent_action)

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = self.state['request_slots']
        user_response['inform_slots'] = self.state['inform_slots']

        reward = self._reward_function(succ)

        return user_response, reward, done, succ

    def _reward_function(self, succ):
        # Todo: Make sure succ = None doesnt trigger False!
        if succ == False:
            reward = -self.max_round
        elif succ == True:
            reward = 2*self.max_round
        else:
            reward = -1
        return reward

    def response_to_request(self, agent_action):
        # First Case: if agent requests for something that is in the usersims goal inform slots, then inform it
        if

        # Second Case: if the agent requests for something in usersims goal request slots and it has already been
        # informed, then inform it

        # Third Case: if the agent requests for something in the usersims goal request slots and it HASNT been
        # informed, then request it with all available inform slots left for usersim to inform

        # Fourth and Final Case: otherwise the usersim does not care about the slot being requested, then inform
        # todo.... fill in

    def response_to_inform(self, agent_action):
        # First Case: If agent informs something that is in goal informs and the value it informed doesnt match, then inform the correct value

        # Second Case: Otherwise pick a random action to take
            # - If anything in state requests then request it
            # - Else if something to say in rest slots, say it
            # - Otherwise respond with 'nothing to say' intent

    def response_to_match_found(self, agent_action):

    def response_to_done(self, agent_action):
