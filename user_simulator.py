import random, copy
from dialogue_config import usersim_default_key, FAIL, NO_OUTCOME, SUCCESS, usersim_required_init_inform_keys, no_query_keys
import random


class UserSimulator:
    def __init__(self, goal_list, constants, database):
        self.goal_list = goal_list
        self.max_round = constants['run']['max_round_num']
        # This is eqivalent to ticket in the moviedatabase, it MUST be in req of user sim goal
        self.default_key = usersim_default_key
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = usersim_required_init_inform_keys
        self.no_query = no_query_keys

        # TEMP ----
        self.database = database
        # ---------

    def reset(self):
        self.goal = random.choice(self.goal_list)
        # Add default slot to requests of goal
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state = {}
        # Add all inform slots informed by agent or usersim to this dict
        self.state['history_slots'] = {}
        # Any inform slots for the current usersim action, empty at start of turn
        self.state['inform_slots'] = {}
        # Current request slots the usersim wants to request
        self.state['request_slots'] = {}
        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state['rest_slots'] = {}
        # Rest slots is going to be a dict of all goal informs and requests
        self.state['rest_slots'].update(self.goal['inform_slots'])
        self.state['rest_slots'].update(self.goal['request_slots'])
        self.state['intent'] = ''
        # False for failure, true for success, auto init to failure
        self.constraint_check = FAIL

        # print('Goal: {}'.format(self.goal))

        return self._return_init_action()

    def _return_init_action(self):
        # Always request
        self.state['intent'] = 'request'

        if self.goal['inform_slots']:
            # So pick all the required init informs, and add if they exist in goal inform slots
            for inform_key in self.init_informs:
                if inform_key in self.goal['inform_slots']:
                    self.state['inform_slots'][inform_key] = self.goal['inform_slots'][inform_key]
                    self.state['rest_slots'].pop(inform_key)
                    self.state['history_slots'][inform_key] = self.goal['inform_slots'][inform_key]
            # If nothing was added then pick a random one to add
            if not self.state['inform_slots']:
                key, value = random.choice(list(self.goal['inform_slots'].items()))
                self.state['inform_slots'][key] = value
                self.state['rest_slots'].pop(key)
                self.state['history_slots'][key] = value

        # Now add a request, do a random one if something other than ticket available
        self.goal['request_slots'].pop(self.default_key)
        if self.goal['request_slots']:
            req_key = random.choice(list(self.goal['request_slots'].keys()))
        else:
            req_key = self.default_key
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state['request_slots'][req_key] = 'UNK'

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])

        # print('Init user action: {}'.format(user_response))

        return user_response

    def step(self, agent_action, round_num):
        # print('ROUND: {}'.format(round_num))
        # print('agent action: {}'.format(agent_action))
        # Assetions
        # No unk in agent action informs
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        # No PLACHEOLDER in agent at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'

        self.state['inform_slots'].clear()
        self.state['intent'] = ''

        done = False
        succ = NO_OUTCOME
        # First check round num, if past max then fail
        if round_num > self.max_round:
            done = True
            succ = FAIL
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            # print('Response:')
            agent_intent = agent_action['intent']
            if agent_intent == 'request':
                # print('req')
                self._response_to_request(agent_action)
            elif agent_intent == 'inform':
                # print('inf')
                self._response_to_inform(agent_action)
            elif agent_intent == 'match_found':
                # print('mf')
                self._response_to_match_found(agent_action)
            elif agent_intent == 'done':
                # print('done')
                succ = self._response_to_done()
                self.state['intent'] = 'done'
                self.state['request_slots'].clear()
                done = True

        # print('Done: {}, Succ.: {}, Constraint Check: {}'.format(done, succ, self.constraint_check))
        # print('State: {}'.format(self.state))

        # My assumptions -------
        # If request intent, then make sure request slots
        if self.state['intent'] == 'request':
            assert self.state['request_slots']
        # If inform intent, then make sure inform slots and NO request slots
        if self.state['intent'] == 'inform':
            assert self.state['inform_slots']
            assert not self.state['request_slots']
        assert 'UNK' not in self.state['inform_slots'].values()
        assert 'PLACEHOLDER' not in self.state['request_slots'].values()
        # No overlap between rest and hist
        for key in self.state['rest_slots']:
            assert key not in self.state['history_slots']
        for key in self.state['history_slots']:
            assert key not in self.state['rest_slots']
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal['inform_slots']:
            assert self.state['history_slots'].get(inf_key, False) or self.state['rest_slots'].get(inf_key, False)
        for req_key in self.goal['request_slots']:
            assert self.state['history_slots'].get(req_key, False) or self.state['rest_slots'].get(req_key, False), req_key
        # Anything in the rest should be in the goal
        for key in self.state['rest_slots']:
            assert self.goal['inform_slots'].get(key, False) or self.goal['request_slots'].get(key, False)
        assert self.state['intent'] != ''
        # -----------------------

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])

        reward = self._reward_function(succ)

        return user_response, reward, done, True if succ is 1 else False

    def _reward_function(self, succ):
        reward = -1
        if succ == FAIL:
            reward += -self.max_round
        elif succ == SUCCESS:
            reward += 2*self.max_round
        return reward

    def _response_to_request(self, agent_action):
        agent_request_key = list(agent_action['request_slots'].keys())[0]
        # First Case: if agent requests for something that is in the usersims goal inform slots, then inform it
        if agent_request_key in self.goal['inform_slots']:
            # print('req 1')
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
            self.state['request_slots'].clear()
            self.state['rest_slots'].pop(agent_request_key, None)
            self.state['history_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
        # Second Case: if the agent requests for something in usersims goal request slots and it has already been
        # informed, then inform it
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['history_slots']:
            # print('req 2')
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.state['history_slots'][agent_request_key]
            self.state['request_slots'].clear()
            assert agent_request_key not in self.state['rest_slots']
        # Third Case: if the agent requests for something in the usersims goal request slots and it HASN'T been
        # informed, then request it with all available inform slots left for usersim to inform
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['rest_slots']:
            # print('req 3')
            # Todo: This was a qualitatve change, so see if it it matters when i converse with agent
            # CHANGED FROm :::
            # self.state['intent'] = 'request'
            # self.state['request_slots'][agent_request_key] = 'UNK'
            # for key in list(self.state['rest_slots'].keys()):
            #     value = self.state['rest_slots'][key]
            #     # Means it is an inform
            #     if value != 'UNK':
            #         self.state['inform_slots'][key] = value
            #         self.state['rest_slots'].pop(key)
            #         self.state['history_slots'][key] = value
            # To:: Clear reqs and add a random inform
            self.state['request_slots'].clear()
            self.state['intent'] = 'request'
            self.state['request_slots'][agent_request_key] = 'UNK'
            rest_informs = {}
            for key, value in list(self.state['rest_slots'].items()):
                if value != 'UNK':
                    rest_informs[key] = value
            if rest_informs:
                key_choice, value_choice = random.choice(list(rest_informs.items()))
                self.state['inform_slots'][key_choice] = value_choice
                self.state['rest_slots'].pop(key_choice)
                self.state['history_slots'][key_choice] = value_choice

        # Fourth and Final Case: otherwise the usersim does not care about the slot being requested, then inform
        else:
            # print('req 4')
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = 'anything'
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_request_key] = 'anything'

    def _response_to_inform(self, agent_action):
        agent_inform_key = list(agent_action['inform_slots'].keys())[0]
        agent_inform_value = agent_action['inform_slots'][agent_inform_key]

        assert agent_inform_key != self.default_key

        # Add all informs (by agent too) to hist slots
        self.state['history_slots'][agent_inform_key] = agent_inform_value
        # Remove from rest slots if in it
        self.state['rest_slots'].pop(agent_inform_key, None)
        # Remove from request slots if in it
        self.state['request_slots'].pop(agent_inform_key, None)

        # First Case: If agent informs something that is in goal informs and the value it informed doesnt match, then inform the correct value
        if agent_inform_value != self.goal['inform_slots'].get(agent_inform_key, agent_inform_value):
            # print('inf 1')
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
        # Second Case: Otherwise pick a random action to take
        else:
            # print('inf 2')
            # - If anything in state requests then request it
            if self.state['request_slots']:
                # print('inf 2.1')
                self.state['intent'] = 'request'
            # - Else if something to say in rest slots, pick something
            elif self.state['rest_slots']:
                # print('inf 2.2')
                # Will return False if not in rest slots, and the value of 'UNK' if it is
                def_in = self.state['rest_slots'].pop(self.default_key, False)
                if self.state['rest_slots']:
                    # print('inf 2.2.1')
                    key, value = random.choice(list(self.state['rest_slots'].items()))
                    # print('Rand choice: {}: {}'.format(key, value))
                    if value != 'UNK':
                        # print('inf 2.2.1.1')
                        self.state['intent'] = 'inform'
                        self.state['inform_slots'][key] = value
                        self.state['rest_slots'].pop(key)
                        self.state['history_slots'][key] = value
                    else:
                        # print('inf 2.2.1.2')
                        self.state['intent'] = 'request'
                        self.state['request_slots'][key] = 'UNK'
                else:
                    # print('inf 2.2.2')
                    self.state['intent'] = 'request'
                    self.state['request_slots'][self.default_key] = 'UNK'
                # If it was in, then add it back because it can only be removed from an inform
                if def_in == 'UNK':
                    self.state['rest_slots'][self.default_key] = 'UNK'
            # - Otherwise respond with 'nothing to say' intent
            else:
                # print('inf 2.3')
                # Note: This thanks will actually have no requests
                self.state['intent'] = 'thanks'

    # All ST informs will be sent in with this agent action
    def _response_to_match_found(self, agent_action):
        # print('mf')
        agent_informs = agent_action['inform_slots']

        # Requests are allowed with this thanks
        self.state['intent'] = 'thanks'
        self.constraint_check = SUCCESS

        # Todo: -- TESTED  --- Called 'add ticket' In mine, remove this clase and instead add match to hist (remove from req and rest) no matter what
        # CHANGED -------- FROM:
        # if agent_informs[self.default_key] == 'no match available':
        #     self.state['history_slots'][self.default_key] = 'no match available'
        #     if self.default_key in self.state['rest_slots']: self.state['rest_slots'].pop(self.default_key)
        #     if self.default_key in self.state['request_slots'].keys(): self.state['request_slots'].pop(self.default_key)
        # To:
        assert self.default_key in agent_informs
        self.state['rest_slots'].pop(self.default_key, None)
        self.state['history_slots'][self.default_key] = str(agent_informs[self.default_key])
        self.state['request_slots'].pop(self.default_key, None)
        # ----------

        # Check if the ticket is not no value match
        if agent_informs[self.default_key] == 'no match available':
            self.constraint_check = FAIL

        # print('agent: {}'.format(agent_informs))
        # print('goal informs: {}'.format(self.goal['inform_slots']))

        # Check to see if all goal informs are in the agent informs, and that the values match
        for key, value in self.goal['inform_slots'].items():
            assert value != None
            # For items that cannot be in the querys dont check to see if they are in the agent informs here
            if key in self.no_query:
                continue
            # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
            if value != agent_informs.get(key, None):
                self.constraint_check = FAIL
                # print('FAIL')
                break

        if self.constraint_check == FAIL:
            self.state['intent'] = 'reject'
            self.state['request_slots'].clear()

        # if self.constraint_check == SUCCESS:
        #     print('----------')
        #     print('agent: {}'.format(agent_action))
        #     print('goal: {}'.format(self.goal))
        #     print('state: {}'.format(self.state))
        #     print('comnstaint: {}'.format(self.constraint_check))

    def _response_to_done(self):
        if self.constraint_check == FAIL:
            return FAIL

        if not self.state['rest_slots']:
            assert not self.state['request_slots']
        if self.state['rest_slots']:
            return FAIL

        # TEMP: ----

        assert self.state['history_slots'][self.default_key] != 'no match available'

        match = copy.deepcopy(self.database[int(self.state['history_slots'][self.default_key])])

        for key, value in self.goal['inform_slots'].items():
            assert value != None
            if key in self.no_query:
                continue
            # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
            if value != match.get(key, None):
                assert True is False, 'match: {}\ngoal: {}'.format(match, self.goal)
                break

        # ----------

        # print('---------------')
        # print('goal: {}'.format(self.goal))
        # print('state: {}'.format(self.state))
        # print('ticket: {}'.format(match))

        return SUCCESS

    # def response_to_done(self, st_informs):
    #     # print('done 1')
    #     # Case 1: Check constraits from match found
    #     if self.constraint_check == FAIL:
    #         # print('constraint fail')
    #         return FAIL
    #
    #     # Case 2: Check if requests and rests empty
    #     # Todo: So they remove the ticket slot before see if rest are empty... idk why, if it doesnt affect performance then remove this
    #     # Will return False if not in rest slots, and the value of 'UNK' if it is
    #     # Assumption is if the rests are empty then so are the requests
    #     # CHANGED FROM: ---------------
    #     # if not self.state['rest_slots']:
    #     #     assert not self.state['request_slots']
    #     # def_in = self.state['rest_slots'].pop(self.default_key, False)
    #     # return_here = False
    #     # if self.state['rest_slots']:
    #     #     return_here = True
    #     # if def_in == 'UNK':
    #     #     self.state['rest_slots'][self.default_key] = 'UNK'
    #     # if return_here:
    #     #     # print('empty fail')
    #     #     return FAIL
    #     # TO:
    #     if not self.state['rest_slots']:
    #         assert not self.state['request_slots']
    #     if self.state['rest_slots']:
    #         return FAIL
    #     # -----------------------------
    #
    #     # Case 3: Check if hist slots contain any NO VALUE MATCH
    #     for key, value in self.state['history_slots'].items():
    #         # TODO: REMOVING CLAUSe----- JK if i do this then i must check that ticket is a VALUE MATCH
    #         # if any no value match in constraints then fail
    #         if value == 'no match available':
    #             return FAIL
    #         # --------------------------
    #         # Todo: Havent gotten to test this (hasnt happened)
    #         if key in self.goal['inform_slots']:
    #             # If informs given by agent/history do not match the goal contraints then fail
    #             if value != self.goal['inform_slots'][key]:
    #                 # print('wrong value fail')
    #                 return FAIL
    #
    #     # Todo: VERYYYYYYYYYYYYYYYYYYYYYYYY IMPORTANT:
    #     # The only reason i bleive that they ONLY add ticket if no match avail (in intent match found) and pop the ticket
    #     # when checking if the rest stack is empty in done is because they check the hist slots to see if no match avail
    #
    #
    #     assert self.state['history_slots'][self.default_key] == 'match available'
    #
    #     # This checks if the final info is wrong....
    #     for key, value in self.goal['inform_slots'].items():
    #         assert value != None
    #         # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
    #         if value != st_informs.get(key, None):
    #             print('\n----- HTTTTT -------\nGoal Informs: {}\nST Informs: {}'.format(self.goal['inform_slots'], st_informs))
    #             break
    #     # ------------
    #
    #     return SUCCESS